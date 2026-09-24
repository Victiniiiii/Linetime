#include "reconcile.h"
#include "utils.h"
#include "lyrics.h"

#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <tuple>

// ---------------------------------------------------------------------------
// Word similarity over normalized strings
// ---------------------------------------------------------------------------
static inline double word_sim(const std::string& a, const std::string& b) {
    return utils::similarity(utils::normalize(a), utils::normalize(b));
}

// ---------------------------------------------------------------------------
// Whisper word preprocessing
// ---------------------------------------------------------------------------
struct WWord {
    std::string text;
    std::string norm;
    double start_ms;
    double end_ms;
    float prob;
    int seg;
};

static std::vector<WWord> clean_whisper_words(const TranscriptionResult& whisper) {
    std::vector<WWord> out;
    int seg_i = 0;
    for (const auto& seg : whisper.segments) {
        for (const auto& w : seg.words) {
            std::string norm = utils::normalize(w.text);
            if (norm.empty()) continue;
            bool alpha = false;
            for (char c : norm) if (std::isalpha((unsigned char)c)) { alpha = true; break; }
            if (!alpha) continue;
            if (w.prob < 0.20f) continue;
            WWord ww;
            ww.text = w.text;
            ww.norm = norm;
            ww.start_ms = w.start_ms;
            ww.end_ms = w.end_ms;
            ww.prob = w.prob;
            ww.seg = seg_i;
            // collapse consecutive identical words at (nearly) the same instant
            if (!out.empty() && out.back().norm == norm &&
                std::abs(out.back().start_ms - w.start_ms) < 40.0 &&
                (w.end_ms - w.start_ms) < 60.0) {
                continue;
            }
            out.push_back(std::move(ww));
        }
        seg_i++;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Main reconcile
// ---------------------------------------------------------------------------
ReconcileResult reconcile_lyrics(const LyricsDocument& hints,
                                 const TranscriptionResult& whisper,
                                 float similarity_threshold) {
    ReconcileResult result;
    (void)similarity_threshold; // text always from hint; whisper only supplies timing

    // 1) Clean whisper words
    std::vector<WWord> wwords = clean_whisper_words(whisper);
    if (wwords.empty()) {
        result.error = "No whisper words";
        return result;
    }
    int WN = (int)wwords.size();

    // 2) Hint lines with normalized text + word offsets
    struct HLine {
        int line_idx;
        std::string text;
        std::string norm;
        std::vector<std::string> words;
        int n_words;
    };
    std::vector<HLine> hlines;
    for (size_t li = 0; li < hints.lines.size(); li++) {
        const auto& line = hints.lines[li];
        if (line.is_ref) continue;
        std::string norm = utils::normalize(line.text);
        if (norm.empty()) continue;
        auto words = utils::split_words(norm);
        if (words.empty()) continue;
        HLine hl;
        hl.line_idx = (int)li;
        hl.text = line.text;
        hl.norm = norm;
        hl.words = words;
        hl.n_words = (int)words.size();
        hlines.push_back(std::move(hl));
    }
    if (hlines.empty()) {
        result.error = "No hint words";
        return result;
    }
    size_t N = hlines.size();

    // 3) Flatten hint words in order
    std::vector<std::string> hint_all;
    std::vector<int> hint_line; // hint word -> line index
    for (size_t j = 0; j < N; j++)
        for (const auto& w : hlines[j].words) {
            hint_all.push_back(w);
            hint_line.push_back((int)j);
        }
    size_t HW = hint_all.size();

    // Diagonal-warp prior: prefer matching hint word i to whisper word j near
    // the linearly-interpolated expected time over the whisper total duration.
    const double WARP = 0.15;
    const double SKIPW = -0.35; // skip a whisper word
    const double SKIPH = -0.50; // skip a hint word

    double t0 = wwords.front().start_ms;
    double t1 = wwords.back().end_ms;
    double span = std::max(t1 - t0, 1.0);
    std::vector<double> expT(HW);
    for (size_t i = 0; i < HW; i++)
        expT[i] = t0 + span * (double)i / (double)std::max<size_t>(HW - 1, 1);
    std::vector<double> wstart(WN);
    for (int j = 0; j < WN; j++) wstart[j] = wwords[j].start_ms;

    // 4) Global word-level DP (Needleman-Wunsch with warp bonus on diagonal)
    std::vector<double> prev(WN + 1, -1e18), curr(WN + 1, 0.0);
    // move[i][j]: 0=diag(match), 1=skip hint word, 2=skip whisper word
    std::vector<std::vector<unsigned char>> move(HW + 1, std::vector<unsigned char>(WN + 1, 0));
    prev[0] = 0;
    for (int j = 1; j <= WN; j++) prev[j] = prev[j - 1] + SKIPW;

    for (size_t i = 1; i <= HW; i++) {
        curr[0] = prev[0] + SKIPH;
        move[i][0] = 1;
        double eT = expT[i - 1];
        const std::string& hi = hint_all[i - 1];
        for (int j = 1; j <= WN; j++) {
            double sim = word_sim(hi, wwords[j - 1].norm);
            double delta_t = std::abs(eT - wstart[j - 1]);
            sim += WARP * std::max(0.0, 1.0 - delta_t / span);
            double diag = prev[j - 1] + sim;
            double down = prev[j] + SKIPH;        // skip hint word
            double right = curr[j - 1] + SKIPW;   // skip whisper word
            if (diag >= down && diag >= right) { curr[j] = diag; move[i][j] = 0; }
            else if (down >= right) { curr[j] = down; move[i][j] = 1; }
            else { curr[j] = right; move[i][j] = 2; }
        }
        prev.swap(curr);
    }

    // 5) Backtrack: collect matched whisper word indices per hint line
    //    matched_sim[line] = sim of each matched hint word; matched_widx[line]
    std::vector<std::vector<int>> wmatch_idx(N);
    std::vector<std::vector<double>> wmatch_sim(N);
    {
        int i = (int)HW, j = WN;
        int safety = 0;
        while (i > 0 && j > 0 && safety < HW + WN + 8) {
            unsigned char m = move[i][j];
            if (m == 0) {
                double sim = word_sim(hint_all[i - 1], wwords[j - 1].norm);
                int li = hint_line[i - 1];
                wmatch_idx[li].push_back(j - 1);
                wmatch_sim[li].push_back(sim);
                i--; j--;
            } else if (m == 1) {
                i--;
            } else {
                j--;
            }
            safety++;
        }
    }

    // 6) Per-line timing + confidence
    std::vector<bool> matched(N, false);
    struct LineTime { double start_ms, end_ms; };
    std::vector<LineTime> lt(N, {-1, -1});
    for (size_t j = 0; j < N; j++) {
        if (wmatch_idx[j].empty()) continue;
        std::vector<int> idx = wmatch_idx[j];
        std::sort(idx.begin(), idx.end());
        // "good" matches (confident word alignment)
        int good = 0;
        for (size_t k = 0; k < idx.size(); k++) if (wmatch_sim[j][k] >= 0.60) good++;
        double coverage = (double)good / (double)hlines[j].n_words;
        bool trust = coverage >= 0.40;
        // line span = first..last matched whisper word
        int f = idx.front(), l = idx.back();
        lt[j].start_ms = wwords[f].start_ms;
        lt[j].end_ms = wwords[l].end_ms;
        if (trust) {
            matched[j] = true;
        } else if (coverage >= 0.15) {
            // weak match: keep whisper timing but low confidence
            matched[j] = false;
        } else {
            lt[j].start_ms = -1;
            lt[j].end_ms = -1;
        }
    }

    // 7) Interpolate unmatched lines between matched (trusted) neighbors
    int first_match = -1;
    for (size_t j = 0; j < N && first_match < 0; j++) if (matched[j]) first_match = (int)j;
    if (first_match < 0) {
        // nothing trusted matched at all -> fall back to whisper segments
        for (const auto& seg : whisper.segments) {
            if (seg.words.empty()) continue;
            ReconciledLine rl;
            rl.text = seg.text;
            rl.start_ms = seg.start_ms;
            rl.end_ms = seg.end_ms;
            rl.from_hint = false;
            rl.confidence = 0.3f;
            result.lines.push_back(rl);
        }
        result.success = true;
        return result;
    }

    // interpolate leading unmatched
    for (int j = first_match - 1; j >= 0; j--) {
        double back = (hlines[j].n_words > 0) ? hlines[j].n_words * 700.0 : 1500.0;
        lt[j].start_ms = std::max(0.0, lt[j + 1].start_ms - back);
        lt[j].end_ms = lt[j + 1].start_ms - 20.0;
        if (lt[j].end_ms < lt[j].start_ms) lt[j].end_ms = lt[j].start_ms;
    }
    // in-between unmatched runs
    {
        int run_start = -1;
        for (int j = 0; j <= (int)N; j++) {
            bool is_match = (j < (int)N) && matched[j];
            if (!is_match && run_start < 0) run_start = j;
            if (is_match && run_start >= 0) {
                int run_end = j - 1;
                double t_end_prev = (lt[run_start - 1].end_ms > 0)
                                        ? lt[run_start - 1].end_ms
                                        : lt[run_start - 1].start_ms;
                double t1 = lt[j].start_ms;
                int total = 0;
                for (int k = run_start; k <= run_end; k++) total += hlines[k].n_words;
                if (total <= 0) total = 1;
                double acc = 0;
                for (int k = run_start; k <= run_end; k++) {
                    acc += hlines[k].n_words;
                    double frac = acc / total;
                    lt[k].start_ms = t_end_prev + (t1 - t_end_prev) * frac;
                    lt[k].end_ms = (k < run_end) ? lt[k + 1].start_ms - 20.0 : t1 - 20.0;
                    if (lt[k].end_ms < lt[k].start_ms) lt[k].end_ms = lt[k].start_ms;
                }
                run_start = -1;
            }
        }
        // trailing unmatched
        if (run_start >= 0) {
            double t_prev = (lt[run_start - 1].end_ms > 0)
                                ? lt[run_start - 1].end_ms
                                : lt[run_start - 1].start_ms;
            for (int k = run_start; k < (int)N; k++) {
                int wc = std::max(hlines[k].n_words, 1);
                lt[k].start_ms = t_prev + 0.05 * 1000;
                lt[k].end_ms = lt[k].start_ms + wc * 550.0;
                t_prev = lt[k].end_ms;
            }
        }
    }

    // 8) Build output lines
    for (size_t j = 0; j < N; j++) {
        ReconciledLine rl;
        rl.text = hlines[j].text;
        rl.from_hint = true;
        if (lt[j].start_ms < 0) lt[j].start_ms = 0;
        if (lt[j].end_ms < 0) lt[j].end_ms = lt[j].start_ms + 1000;
        rl.start_ms = lt[j].start_ms;
        rl.end_ms = lt[j].end_ms;
        rl.confidence = matched[j] ? 0.8f : 0.4f;
        result.lines.push_back(rl);
    }

    // 9) Enforce monotonic non-decreasing times
    std::vector<ReconciledLine> final_lines;
    for (size_t i = 0; i < result.lines.size(); i++) {
        ReconciledLine rl = result.lines[i];
        if (rl.start_ms > rl.end_ms) std::swap(rl.start_ms, rl.end_ms);
        if (!final_lines.empty() && rl.start_ms < final_lines.back().end_ms) {
            rl.start_ms = final_lines.back().end_ms;
        }
        if (rl.end_ms < rl.start_ms) rl.end_ms = rl.start_ms;
        final_lines.push_back(rl);
    }

    // 10) Paragraph breaks when gap large
    result.lines.clear();
    for (size_t i = 0; i < final_lines.size(); i++) {
        result.lines.push_back(final_lines[i]);
        if (i + 1 < final_lines.size()) {
            double gap = final_lines[i + 1].start_ms - final_lines[i].end_ms;
            if (gap > 2000) {
                ReconciledLine blank;
                blank.text = "";
                blank.start_ms = final_lines[i].end_ms;
                blank.end_ms = final_lines[i].end_ms;
                blank.from_hint = false;
                blank.confidence = 0;
                result.lines.push_back(blank);
            }
        }
    }

    result.success = true;
    return result;
}
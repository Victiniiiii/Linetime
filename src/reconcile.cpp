#include "reconcile.h"
#include "utils.h"
#include "lyrics.h"

#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <tuple>
#include <cstdio>

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

// Whisper tokens that describe the audio rather than lyrics (hallucination markers)
static inline bool is_whisper_marker(const std::string& w) {
    std::string lo = utils::normalize(w);
    return lo == "music" || lo == "applause" || lo == "laughing" ||
           lo == "laughter" || lo == "applauding" || lo == "singing";
}

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
                                 float similarity_threshold,
                                 bool recover_missing) {
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
        // Whisper matched nothing confidently — i.e. the transcript is likely a
        // hallucination (instrumental intro, chant loops, "subtitles by…" mode).
        // Discarding the hint for whisper segments here is catastrophic: the
        // authored lyrics *are* the authority, and our CTC refinement can align
        // them to the audio directly (method-A quality). Emit the hint lines on
        // a neutral linear spread over the whisper span and let CTC re-time.
        fprintf(stderr, "  [reconcile] whisper unreliably matched no hint line; "
                        "falling back to hint + CTC alignment\n");
        double t0w = WN ? wwords.front().start_ms : 0.0;
        double t1w = WN ? wwords.back().end_ms : 0.0;
        if (t1w <= t0w) t1w = t0w + 1000.0;
        double span = t1w - t0w;
        for (size_t j = 0; j < N; j++) {
            double f = (double)j / (double)std::max(N, (size_t)1);
            double f2 = (double)(j + 1) / (double)std::max(N, (size_t)1);
            lt[j].start_ms = t0w + f * span;
            lt[j].end_ms = t0w + f2 * span;
            matched[j] = false;
            wmatch_idx[j].clear();
            wmatch_sim[j].clear();
        }
        // skip fill (recovered) pass: whisper words are untrustworthy
        recover_missing = false;
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

    // 8) Build output lines. Reconstruct line text from the whisper words the
    //    DP attached to it (from first..last confident match, including interior
    //    words whisper heard but the hint spelled differently or dropped) —
    //    this fixes typos and restores words. Only confident matches (sim>=0.5)
    //    mark whisper coverage, so words the DP matched weakly to the wrong
    //    line stay uncovered and can be re-emitted by the fill pass below.
    std::vector<bool> covered(WN, false);
    auto trim_space = [](std::string& s) {
        size_t a = s.find_first_not_of(' '), b = s.find_last_not_of(' ');
        s = (a == std::string::npos) ? std::string() : s.substr(a, b - a + 1);
    };
    result.lines.clear();
    for (size_t j = 0; j < N; j++) {
        ReconciledLine rl;
        std::vector<int> anchors;
        std::vector<int> idx = wmatch_idx[j];
        for (size_t k = 0; k < idx.size(); k++)
            if (wmatch_sim[j][k] >= 0.50) anchors.push_back(idx[k]);
        std::sort(anchors.begin(), anchors.end());
        if (matched[j]) {
            // Hint text is trustworthy (high coverage against whisper): keep it
            // verbatim — whisper's spelling of a well-matched line (e.g. a
            // misheard vowel or a romanized name) is usually worse than the hint.
            rl.text = hlines[j].text;
            rl.align_text = hlines[j].text;
            rl.from_hint = true;
            rl.confidence = 0.8f;
            rl.start_ms = lt[j].start_ms;
            rl.end_ms = lt[j].end_ms;
            if (rl.start_ms < 0) rl.start_ms = 0;
            if (rl.end_ms < 0) rl.end_ms = rl.start_ms + 1000;
            // only confident anchors mark whisper coverage
            for (int k : anchors) covered[k] = true;
        } else if (!anchors.empty() && lt[j].start_ms >= 0) {
            int f = anchors.front(), l = anchors.back();
            rl.start_ms = wwords[f].start_ms;
            rl.end_ms = wwords[l].end_ms;
            // Rebuild from whisper when the hint line is weak: fixes typos /
            // dropped words. Include interior whisper words only when the
            // resulting span is plausibly a single line.
            std::string txt;
            double span_ms = wwords[l].end_ms - wwords[f].start_ms;
            bool include_interiors = span_ms <= (anchors.size() + 2) * 600.0;
            int prev_k = -1;
            for (int k = f; k <= l; k++) {
                bool anchor = std::binary_search(anchors.begin(), anchors.end(), k);
                if (is_whisper_marker(wwords[k].text)) continue;
                if (!anchor) {
                    if (!include_interiors) continue;
                    if (prev_k < 0) continue;
                    if (wwords[k].start_ms - wwords[prev_k].end_ms >= 900.0) continue;
                }
                txt += ' ';
                txt += wwords[k].text;
                prev_k = k;
                covered[k] = true;
            }
            trim_space(txt);
            rl.text = txt;
            rl.align_text = hlines[j].text; // CTC anchors on hint spelling
            rl.from_hint = false;
            rl.confidence = 0.4f;
            for (int k : anchors) covered[k] = true;
        } else {
            rl.text = hlines[j].text;
            rl.align_text.clear();
            rl.from_hint = true;
            rl.confidence = 0.3f;
            rl.start_ms = lt[j].start_ms;
            rl.end_ms = lt[j].end_ms;
            if (rl.start_ms < 0) rl.start_ms = 0;
            if (rl.end_ms < 0) rl.end_ms = rl.start_ms + 1000;
        }
        result.lines.push_back(std::move(rl));
    }

    // 8.5) Recover sung regions the hint omitted entirely (e.g. a chorus the
    //      lyrics file does not repeat) — OFF by default since whisper
    //      hallucination chant-tails would otherwise be re-emitted as lyrics.
    //      When enabled, whisper words that no hint line anchored form
    //      uncovered runs; emit them as lines (mirroring the hint's own line
    //      structure when the section repeats text the hint keeps).
    if (recover_missing) {
        auto flush = [&](int rs, int re) {
            if (re < rs) return;
            int n = re - rs + 1;
            // only reconstructed sections worth filling: a removed chorus/verse,
            // not mere word-gaps from imperfect DP matching
            if (n < 6) return;
            if (wwords[re].end_ms - wwords[rs].start_ms < 3000.0) return;
            auto emit = [&](std::vector<int> words, int hint_line_idx) {
                if ((int)words.size() < 2) return;
                double avg = 0;
                for (int k : words) avg += wwords[k].prob;
                if (avg / words.size() < 0.45) return;
                std::string txt;
                for (int k : words) {
                    if (is_whisper_marker(wwords[k].text)) continue;
                    txt += ' ';
                    txt += wwords[k].text;
                }
                trim_space(txt);
                if (txt.empty()) return;
                ReconciledLine rl;
                rl.text = txt;
                if (hint_line_idx >= 0 && hlines[hint_line_idx].text.size() > 0)
                    rl.align_text = hlines[hint_line_idx].text; // CTC anchor
                rl.start_ms = wwords[words.front()].start_ms;
                rl.end_ms = wwords[words.back()].end_ms;
                rl.from_hint = false;
                rl.recovered = true;
                rl.confidence = 0.5f; // recovered from audio, no hint anchor
                result.lines.push_back(std::move(rl));
            };
            // Try to mirror the hint's own line structure onto this run: the
            // run is usually a section the hint omitted but repeats verbatim
            // elsewhere (chorus). A local monotonic DP maps run words -> hint
            // words; segments ending on a hint-line boundary become lines, and
            // each line reuses the hint line's text as its CTC alignment anchor.
            std::vector<int> run_txt;
            for (int k = rs; k <= re; k++) run_txt.push_back(k);
            int RP = (int)run_txt.size();
            const double LSKIPW = -0.35, LSKIPH = -0.50;
            std::vector<std::vector<double>> sc(RP + 1, std::vector<double>(HW + 1, -1e18));
            std::vector<std::vector<unsigned char>> mv(RP + 1, std::vector<unsigned char>(HW + 1, 0));
            sc[0][0] = 0;
            for (int j = 1; j <= (int)HW; j++) sc[0][j] = sc[0][j - 1] + LSKIPW;
            for (int i = 1; i <= RP; i++) {
                sc[i][0] = sc[i - 1][0] + LSKIPH;
                mv[i][0] = 1;
                for (int j = 1; j <= (int)HW; j++) {
                    double sim = word_sim(wwords[run_txt[i - 1]].norm, hint_all[j - 1]);
                    double diag = sc[i - 1][j - 1] + sim;
                    double down = sc[i - 1][j] + LSKIPH;
                    double right = sc[i][j - 1] + LSKIPW;
                    if (diag >= down && diag >= right) { sc[i][j] = diag; mv[i][j] = 0; }
                    else if (down >= right) { sc[i][j] = down; mv[i][j] = 1; }
                    else { sc[i][j] = right; mv[i][j] = 2; }
                }
            }
            // backtrack, tagging each run word with its hint line (or -1)
            std::vector<int> tag(RP, -1);
            {
                int i = RP, j = (int)HW;
                while (i > 0 && j > 0) {
                    unsigned char m = mv[i][j];
                    if (m == 0) {
                        tag[i - 1] = hint_line[j - 1];
                        i--; j--;
                    } else if (m == 1) i--;
                    else j--;
                }
            }
            // keep mirror only for a coherent, reasonably-fitting region
            int tagged = 0;
            double acc = 0;
            for (int i = 0; i < RP; i++) {
                if (tag[i] >= 0) { tagged++; acc += 1.0; }
            }
            if (tagged >= std::min(RP, 3) && (double)tagged / RP >= 0.5) {
                int hprev = -1;
                std::vector<int> cur;
                for (int i = 0; i < RP; i++) {
                    int h = tag[i];
                    if (h != hprev && !cur.empty()) { emit(cur, hprev); cur.clear(); }
                    hprev = (h >= 0) ? h : hprev;
                    cur.push_back(run_txt[i]);
                }
                if (!cur.empty()) emit(cur, hprev);
                return;
            }
            // fallback: split at word gaps > 900ms => natural lyric boundaries
            int gs = rs;
            while (gs <= re) {
                int ge = gs;
                while (ge + 1 <= re &&
                       wwords[ge + 1].start_ms - wwords[ge].end_ms < 900.0)
                    ge++;
                std::vector<int> grp;
                for (int k = gs; k <= ge; k++) grp.push_back(k);
                emit(grp, -1);
                gs = ge + 1;
            }
        };
        int run_start = -1;
        for (int k = 0; k <= WN; k++) {
            if (k < WN && !covered[k]) {
                if (run_start < 0) run_start = k;
            } else if (run_start >= 0) {
                flush(run_start, k - 1);
                run_start = -1;
            }
        }
    }

    // 9) Keep hint lines in their original (lyrics) order and enforce monotonic
    //    non-decreasing times; recovered (fill) lines are inserted by time so
    //    they never reorder the hint lyrics (whisper times on weak tail lines
    //    would otherwise scramble order and break downstream alignment).
    std::vector<ReconciledLine> base;
    std::vector<ReconciledLine> recovered;
    for (size_t i = 0; i < result.lines.size(); i++) {
        ReconciledLine rl = result.lines[i];
        if (rl.recovered) { recovered.push_back(std::move(rl)); continue; }
        if (rl.start_ms > rl.end_ms) std::swap(rl.start_ms, rl.end_ms);
        if (!base.empty() && rl.start_ms < base.back().end_ms) {
            rl.start_ms = base.back().end_ms;
        }
        if (rl.end_ms < rl.start_ms) rl.end_ms = rl.start_ms;
        base.push_back(std::move(rl));
    }
    std::vector<ReconciledLine> final_lines;
    final_lines.reserve(base.size() + recovered.size());
    // merge recovered lines in time order
    std::sort(recovered.begin(), recovered.end(),
              [](const ReconciledLine& a, const ReconciledLine& b) {
                  return a.start_ms < b.start_ms;
              });
    size_t bi = 0, ri = 0;
    while (bi < base.size() || ri < recovered.size()) {
        if (bi < base.size() && (ri >= recovered.size() ||
                                 base[bi].start_ms <= recovered[ri].start_ms)) {
            ReconciledLine rl = base[bi++];
            if (!final_lines.empty() && rl.start_ms < final_lines.back().end_ms)
                rl.start_ms = final_lines.back().end_ms;
            final_lines.push_back(std::move(rl));
        } else {
            ReconciledLine rl = recovered[ri++];
            if (!final_lines.empty() && rl.start_ms < final_lines.back().end_ms)
                rl.start_ms = final_lines.back().end_ms;
            final_lines.push_back(std::move(rl));
        }
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
                blank.recovered = false;
                blank.confidence = 0;
                result.lines.push_back(blank);
            }
        }
    }

    result.success = true;
    return result;
}
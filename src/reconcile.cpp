#include "reconcile.h"
#include "utils.h"
#include "lyrics.h"

#include <algorithm>
#include <vector>
#include <string>
#include <cmath>

static std::vector<std::string> split_words_norm(const std::string& s) {
    std::string n = utils::normalize(s);
    return utils::split_words(n);
}

static double word_similarity(const std::string& a, const std::string& b) {
    return utils::similarity(utils::normalize(a), utils::normalize(b));
}

// Word-level alignment: align hint words to whisper words
struct AlignedWord {
    int w_idx;        // whisper word index
    int h_idx;        // hint word index
    double similarity;
};

static std::vector<AlignedWord> align_words(const std::vector<std::string>& whisper_norms,
                                            const std::vector<std::string>& hint_norms) {
    int W = whisper_norms.size();
    int H = hint_norms.size();
    if (W == 0 || H == 0) return {};

    // DP: dp[i][j] = max similarity sum for first i whisper words, first j hint words
    // Moves: match (diag), skip whisper (down), skip hint (right)
    // Score: match = similarity, skip whisper = -0.05, skip hint = -0.1
    std::vector<double> prev(H + 1, -1e9), curr(H + 1);
    std::vector<std::vector<char>> move(W + 1, std::vector<char>(H + 1, 0));

    prev[0] = 0;
    for (int j = 1; j <= H; j++) prev[j] = prev[j-1] - 0.1;

    for (int i = 1; i <= W; i++) {
        curr[0] = prev[0] - 0.05;
        for (int j = 1; j <= H; j++) {
            double sim = word_similarity(whisper_norms[i-1], hint_norms[j-1]);
            double diag = prev[j-1] + sim;
            double down = prev[j] - 0.05;
            double right = curr[j-1] - 0.1;

            if (diag >= down && diag >= right) {
                curr[j] = diag;
                move[i][j] = 0; // match
            } else if (down >= right) {
                curr[j] = down;
                move[i][j] = 1; // skip whisper
            } else {
                curr[j] = right;
                move[i][j] = 2; // skip hint
            }
        }
        prev.swap(curr);
    }

    fprintf(stderr, "[align_words] DP done, backtracking...\n");

    // Backtrack
    std::vector<AlignedWord> alignment;
    int i = W, j = H;
    int safety = 0;
    while ((i > 0 || j > 0) && safety < W + H + 10) {
        if (i < 0) i = 0;
        if (j < 0) j = 0;
        if (i == 0 && j == 0) break;
        char m = (i > 0 && j > 0) ? move[i][j] : (i > 0 ? 1 : 2);
        if (m == 0) {
            double sim = word_similarity(whisper_norms[i-1], hint_norms[j-1]);
            alignment.push_back({i-1, j-1, sim});
            i--; j--;
        } else if (m == 1) {
            i--;
        } else {
            j--;
        }
        safety++;
    }
    if (safety >= W + H + 10) {
        fprintf(stderr, "[align_words] WARNING: backtrack safety limit reached\n");
    }
    std::reverse(alignment.begin(), alignment.end());
    return alignment;
}

ReconcileResult reconcile_lyrics(const LyricsDocument& hints,
                                 const TranscriptionResult& whisper,
                                 float similarity_threshold) {
    ReconcileResult result;

    // Flatten whisper words with timestamps
    struct WWord {
        std::string text;
        std::string norm;
        double start_ms;
        double end_ms;
        float prob;
    };
    std::vector<WWord> wwords;
    for (const auto& seg : whisper.segments) {
        for (const auto& w : seg.words) {
            std::string n = utils::normalize(w.text);
            if (!n.empty()) {
                wwords.push_back({w.text, n, w.start_ms, w.end_ms, w.prob});
            }
        }
    }
    if (wwords.empty()) {
        result.error = "No whisper words";
        return result;
    }

    // Flatten hint words with line back-references
    struct HWord {
        std::string text;
        std::string norm;
        int line_idx;
    };
    std::vector<HWord> hwords;
    std::vector<int> line_start_idx;
    std::vector<int> line_end_idx;
    
    for (size_t li = 0; li < hints.lines.size(); li++) {
        const auto& line = hints.lines[li];
        if (line.is_ref || line.normalized.empty()) continue;
        auto words = split_words_norm(line.text);
        if (words.empty()) continue;
        
        line_start_idx.push_back(hwords.size());
        for (const auto& w : words) {
            hwords.push_back({w, utils::normalize(w), (int)li});
        }
        line_end_idx.push_back(hwords.size() - 1);
    }
    if (hwords.empty()) {
        result.error = "No hint words";
        return result;
    }

    // Safety limit: if too many words, fall back to segment-based approach
    if (wwords.size() > 2000 || hwords.size() > 2000) {
        fprintf(stderr, "[reconcile] Too many words (%zu whisper, %zu hint), using segment fallback\n",
                wwords.size(), hwords.size());
        for (const auto& seg : whisper.segments) {
            if (!seg.words.empty()) {
                ReconciledLine rl;
                rl.text = seg.text;
                rl.start_ms = seg.start_ms;
                rl.end_ms = seg.end_ms;
                rl.from_hint = false;
                rl.confidence = 0.5f;
                result.lines.push_back(rl);
            }
        }
        result.success = true;
        return result;
    }

    // Prepare arrays for alignment
    std::vector<std::string> whisper_norms, hint_norms;
    for (const auto& w : wwords) whisper_norms.push_back(w.norm);
    for (const auto& h : hwords) hint_norms.push_back(h.norm);

    // Align word sequences
    auto alignment = align_words(whisper_norms, hint_norms);
    if (alignment.empty()) {
        result.error = "Alignment failed";
        return result;
    }

    // For each hint line, collect its aligned whisper words
    struct LineMatch {
        int line_idx = -1;
        std::vector<int> w_indices;
        double avg_sim = 0;
    };
    std::vector<LineMatch> line_matches;
    
    // Group alignment by hint line
    std::vector<std::vector<int>> line_to_widx(hints.lines.size());
    for (const auto& aw : alignment) {
        int li = hwords[aw.h_idx].line_idx;
        if (li >= 0 && li < (int)line_to_widx.size()) {
            line_to_widx[li].push_back(aw.w_idx);
        }
    }

    // Build result lines
    for (size_t li = 0; li < hints.lines.size(); li++) {
        const auto& line = hints.lines[li];
        if (line.is_ref || line.normalized.empty()) continue;

        const auto& wids = line_to_widx[li];
        if (wids.empty()) {
            // No alignment for this line - skip
            continue;
        }

        // Compute average similarity
        double sum_sim = 0;
        for (int wid : wids) {
            // Find alignment entry for this wid
            for (const auto& aw : alignment) {
                if (aw.w_idx == wid) {
                    sum_sim += aw.similarity;
                    break;
                }
            }
        }
        double avg_sim = wids.empty() ? 0 : sum_sim / wids.size();

        // Get timestamps from first/last aligned whisper word
        double start_ms = wwords[wids.front()].start_ms;
        double end_ms = wwords[wids.back()].end_ms;

        ReconciledLine rl;
        rl.start_ms = start_ms;
        rl.end_ms = end_ms;
        rl.confidence = (float)avg_sim;

        // Decide text: if similarity high enough, use hint; else reconstruct from whisper
        if (avg_sim >= similarity_threshold) {
            rl.text = line.text;
            rl.from_hint = true;
        } else {
            // Reconstruct from whisper words
            std::string wtext;
            for (int wid : wids) {
                if (!wtext.empty()) wtext += " ";
                wtext += wwords[wid].text;
            }
            rl.text = wtext;
            rl.from_hint = false;
        }
        result.lines.push_back(rl);
    }

    // If too few lines matched, fall back to whisper segments
    if (result.lines.size() < hints.lines.size() * 0.3) {
        result.lines.clear();
        for (const auto& seg : whisper.segments) {
            if (!seg.words.empty()) {
                ReconciledLine rl;
                rl.text = seg.text;
                rl.start_ms = seg.start_ms;
                rl.end_ms = seg.end_ms;
                rl.from_hint = false;
                rl.confidence = 0.5f;
                result.lines.push_back(rl);
            }
        }
    }

    // Post-process: ensure monotonic, add paragraph breaks
    std::vector<ReconciledLine> final_lines;
    for (size_t i = 0; i < result.lines.size(); i++) {
        ReconciledLine rl = result.lines[i];
        if (rl.start_ms > rl.end_ms) std::swap(rl.start_ms, rl.end_ms);
        if (!final_lines.empty() && rl.start_ms < final_lines.back().end_ms) {
            rl.start_ms = final_lines.back().end_ms;
        }
        final_lines.push_back(rl);
    }

    // Add blank lines for paragraph breaks (gap > 1.5s)
    result.lines.clear();
    for (size_t i = 0; i < final_lines.size(); i++) {
        result.lines.push_back(final_lines[i]);
        if (i + 1 < final_lines.size()) {
            double gap = final_lines[i + 1].start_ms - final_lines[i].end_ms;
            if (gap > 1500) {
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
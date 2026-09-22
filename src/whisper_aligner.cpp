#include "whisper_aligner.h"
#include "utils.h"
#include <whisper.h>
#include <algorithm>
#include <cstring>
#include <cmath>

struct WhisperAligner::Impl {
    whisper_context* ctx = nullptr;
    bool initialized = false;
    ~Impl() { if (ctx) whisper_free(ctx); }
};

WhisperAligner::WhisperAligner() : impl_(new Impl()) {}
WhisperAligner::~WhisperAligner() { delete impl_; }

bool WhisperAligner::init(const std::string& model_path, Provider provider) {
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.flash_attn = false;

    if (provider == Provider::CUDA || provider == Provider::Auto) {
        cparams.use_gpu = true;
        cparams.gpu_device = 0;
        fprintf(stderr, "[whisper] Using GPU acceleration (CUDA)\n");
    } else {
        cparams.use_gpu = false;
        fprintf(stderr, "[whisper] Using CPU\n");
    }

    impl_->ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (!impl_->ctx) {
        fprintf(stderr, "[whisper] Failed to load model from %s\n", model_path.c_str());
        return false;
    }
    impl_->initialized = true;
    fprintf(stderr, "[whisper] Model loaded: %s\n", model_path.c_str());
    return true;
}

CTCAlignerResult WhisperAligner::align(const AudioBuffer& audio,
                                        const LyricsDocument& lyrics,
                                        const std::string& language) {
    CTCAlignerResult result;
    if (!impl_->initialized) { result.error = "Whisper not initialized"; return result; }

    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.strategy = WHISPER_SAMPLING_GREEDY;
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;
    wparams.print_special = false;
    wparams.translate = false;
    wparams.single_segment = false;
    wparams.no_timestamps = false;
    wparams.token_timestamps = true;
    wparams.thold_pt = 0.01f;
    wparams.thold_ptsum = 0.01f;
    wparams.max_len = 1;
    wparams.split_on_word = true;
    wparams.n_threads = 4;

    if (language != "auto" && !language.empty()) {
        wparams.language = language.c_str();
    }

    if (whisper_full(impl_->ctx, wparams, audio.samples.data(), audio.n_samples) != 0) {
        result.error = "Whisper inference failed";
        return result;
    }

    int n_segments = whisper_full_n_segments(impl_->ctx);
    fprintf(stderr, "[whisper] Got %d segments\n", n_segments);

    struct SegInfo {
        std::string text;
        std::string normalized;
        int64_t t0_ms;
        int64_t t1_ms;
    };
    std::vector<SegInfo> segments;

    for (int i = 0; i < n_segments; i++) {
        const char* seg_text = whisper_full_get_segment_text(impl_->ctx, i);
        if (!seg_text || seg_text[0] == '\0') continue;

        int64_t t0 = whisper_full_get_segment_t0(impl_->ctx, i) * 10;
        int64_t t1 = whisper_full_get_segment_t1(impl_->ctx, i) * 10;
        std::string text = seg_text;
        std::string norm = utils::normalize(text);
        if (!norm.empty())
            segments.push_back({text, norm, t0, t1});
    }

    fprintf(stderr, "[whisper] Collected %zu segments\n", segments.size());

    if (segments.empty()) {
        result.error = "No segments from whisper";
        return result;
    }

    // Build combined transcript for debugging
    std::string full_transcript;
    for (const auto& seg : segments) full_transcript += seg.text;
    fprintf(stderr, "[whisper] Transcript: %.300s\n", full_transcript.c_str());

    // Collect non-ref lyric lines
    struct LineInfo { int index; std::string text; std::string normalized; };
    std::vector<LineInfo> lines;
    for (const auto& line : lyrics.lines) {
        if (line.is_ref || line.normalized.empty()) continue;
        std::string norm = utils::normalize(line.text);
        if (norm.empty()) continue;
        lines.push_back({line.index, line.text, norm});
    }

    int n_lines = (int)lines.size();
    int n_segs = (int)segments.size();
    int64_t total_duration = segments.back().t1_ms;

    fprintf(stderr, "[whisper] Aligning %d lines to %d segments\n", n_lines, n_segs);

    // Build segment text (concatenate adjacent segments for better matching)
    // Use a sliding window over segments to create "super-segments"
    // Each super-segment covers 1-5 consecutive segments
    struct SuperSeg {
        std::string normalized;
        int64_t t0_ms;
        int64_t t1_ms;
    };
    std::vector<SuperSeg> super_segs;

    for (int span = 1; span <= std::min(5, n_segs); span++) {
        for (int start = 0; start + span <= n_segs; start++) {
            std::string combined;
            for (int j = start; j < start + span; j++)
                combined += segments[j].normalized;
            super_segs.push_back({combined, segments[start].t0_ms, segments[start + span - 1].t1_ms});
        }
    }

    fprintf(stderr, "[whisper] Built %zu super-segments\n", super_segs.size());

    // For each line, find best matching super-segment
    // Use Needleman-Wunsch-style DP for monotonic assignment
    // dp[i][j] = best score for assigning first i lines using segments up to j

    // First, for each line, find its best matching super-segment position (character offset)
    // This gives us a "preferred position" for each line in the combined transcript

    std::string combined_transcript;
    for (const auto& ss : super_segs) combined_transcript += ss.normalized;
    int combined_len = (int)combined_transcript.size();

    // Build super-segment character offsets
    std::vector<int> ss_char_offsets;
    {
        int offset = 0;
        for (const auto& ss : super_segs) {
            ss_char_offsets.push_back(offset);
            offset += ss.normalized.size();
        }
        ss_char_offsets.push_back(offset);
    }

    // For each line, find best position in combined transcript
    struct LinePos {
        float score;
        int char_pos;
        int ss_idx;  // super-segment index
    };

    std::vector<std::vector<LinePos>> line_positions(n_lines);
    for (int i = 0; i < n_lines; i++) {
        const auto& lnorm = lines[i].normalized;
        int line_len = (int)lnorm.size();

        for (int pos = 0; pos + line_len <= combined_len; pos += 3) {
            for (int expand = 0; expand <= std::min(line_len / 3, 10); expand++) {
                int len = line_len + expand;
                if (pos + len > combined_len) break;
                std::string candidate = combined_transcript.substr(pos, len);
                double sim = utils::similarity(lnorm, candidate);
                if (sim > 0.3) {
                    // Find which super-segment this falls in
                    int ss_idx = 0;
                    for (int s = 0; s < (int)super_segs.size(); s++) {
                        if (ss_char_offsets[s] <= pos && pos < ss_char_offsets[s + 1]) {
                            ss_idx = s; break;
                        }
                    }
                    line_positions[i].push_back({(float)sim, pos, ss_idx});
                    break;
                }
            }
        }

        // Sort by score
        std::sort(line_positions[i].begin(), line_positions[i].end(),
            [](const LinePos& a, const LinePos& b) { return a.score > b.score; });

        fprintf(stderr, "[whisper] Line %d: %zu candidates, best=%.3f\n",
                i, line_positions[i].size(),
                line_positions[i].empty() ? 0.0f : line_positions[i][0].score);
    }

    // DP: best monotonic assignment
    // dp[i][k] = best total score for first i+1 lines, with line i at candidate k
    struct DPEntry { float score; int prev_k; };

    std::vector<std::vector<DPEntry>> dp(n_lines);
    std::vector<std::vector<int>> best_prev(n_lines);

    // Initialize first line
    for (int k = 0; k < (int)line_positions[0].size(); k++) {
        dp[0].push_back({line_positions[0][k].score, -1});
    }

    // Fill DP
    for (int i = 1; i < n_lines; i++) {
        dp[i].resize(line_positions[i].size(), {-1e9f, -1});

        for (int k = 0; k < (int)line_positions[i].size(); k++) {
            int cur_pos = line_positions[i][k].char_pos;
            float line_score = line_positions[i][k].score;

            for (int j = 0; j < (int)line_positions[i - 1].size(); j++) {
                int prev_pos = line_positions[i - 1][j].char_pos;
                if (prev_pos < cur_pos) {
                    float total = dp[i - 1][j].score + line_score;
                    if (total > dp[i][k].score) {
                        dp[i][k] = {total, j};
                    }
                }
            }

            // Allow skipping (unmatched) with penalty
            if (dp[i][k].score < 0) {
                dp[i][k] = {line_score - 0.5f, -1};
            }
        }
    }

    // Backtrace
    int best_end_k = 0;
    float best_total = -1e9f;
    for (int k = 0; k < (int)dp[n_lines - 1].size(); k++) {
        if (dp[n_lines - 1][k].score > best_total) {
            best_total = dp[n_lines - 1][k].score;
            best_end_k = k;
        }
    }

    std::vector<int> assignment(n_lines, -1);
    {
        int cur_line = n_lines - 1;
        int cur_k = best_end_k;
        while (cur_line >= 0 && cur_k >= 0 && cur_k < (int)dp[cur_line].size()) {
            assignment[cur_line] = cur_k;
            cur_k = dp[cur_line][cur_k].prev_k;
            cur_line--;
        }
    }

    // Build result
    for (int i = 0; i < n_lines; i++) {
        AlignedLine al;
        al.line_index = lines[i].index;
        al.text = lines[i].text;

        if (assignment[i] >= 0 && assignment[i] < (int)line_positions[i].size()) {
            const auto& pos = line_positions[i][assignment[i]];
            al.confidence = pos.score;
            // Map char position to super-segment timestamps
            al.start_ms = super_segs[pos.ss_idx].t0_ms;
            al.end_ms = super_segs[pos.ss_idx].t1_ms;
        } else {
            // Unmatched — interpolate
            float progress = (float)(i + 1) / (n_lines + 1);
            al.start_ms = (int64_t)(progress * total_duration);
            al.end_ms = al.start_ms;
            al.confidence = 0;
        }

        result.lines.push_back(al);
    }

    result.success = true;
    fprintf(stderr, "[whisper] Aligned %zu lines\n", result.lines.size());
    return result;
}

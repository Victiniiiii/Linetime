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

bool WhisperAligner::init(const std::string& model_path) {
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.flash_attn = false;

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

    // Collect segments with timestamps
    struct SegInfo {
        std::string text;
        int64_t t0_ms;
        int64_t t1_ms;
    };
    std::vector<SegInfo> segments;

    for (int i = 0; i < n_segments; i++) {
        const char* seg_text = whisper_full_get_segment_text(impl_->ctx, i);
        if (!seg_text || seg_text[0] == '\0') continue;

        int64_t t0 = whisper_full_get_segment_t0(impl_->ctx, i); // in centiseconds (10ms)
        int64_t t1 = whisper_full_get_segment_t1(impl_->ctx, i);

        segments.push_back({seg_text, t0 * 10, t1 * 10}); // convert to ms
    }

    fprintf(stderr, "[whisper] Collected %zu segments\n", segments.size());

    if (segments.empty()) {
        result.error = "No segments from whisper";
        return result;
    }

    // Also try to get word-level tokens
    struct TokenInfo {
        std::string text;
        int64_t t0_ms;
        int64_t t1_ms;
    };
    std::vector<TokenInfo> tokens;

    for (int i = 0; i < n_segments; i++) {
        int n_tokens = whisper_full_n_tokens(impl_->ctx, i);
        for (int j = 0; j < n_tokens; j++) {
            whisper_token_data data = whisper_full_get_token_data(impl_->ctx, i, j);
            if (data.id >= whisper_token_eot(impl_->ctx)) continue;
            const char* tok_text = whisper_full_get_token_text(impl_->ctx, i, j);
            if (!tok_text || tok_text[0] == '\0') continue;

            // Use token timestamps if available, else segment timestamps
            int64_t t0 = (data.t0 > 0) ? data.t0 : whisper_full_get_segment_t0(impl_->ctx, i) * 10;
            int64_t t1 = (data.t1 > 0) ? data.t1 : whisper_full_get_segment_t1(impl_->ctx, i) * 10;

            tokens.push_back({tok_text, t0, t1});
        }
    }

    fprintf(stderr, "[whisper] Collected %zu tokens\n", tokens.size());

    // Build full transcript
    std::string full_transcript;
    for (const auto& seg : segments) full_transcript += seg.text;
    fprintf(stderr, "[whisper] Transcript: %.300s\n", full_transcript.c_str());

    // Align each lyric line to transcript segments using DP
    // Concatenate all segment texts and find character offsets
    std::vector<int> seg_char_offsets; // char offset where each segment starts
    {
        int offset = 0;
        for (const auto& seg : segments) {
            seg_char_offsets.push_back(offset);
            offset += seg.text.size();
        }
        seg_char_offsets.push_back(offset); // end sentinel
    }

    int seg_idx = 0; // current segment index for forward scanning

    for (const auto& line : lyrics.lines) {
        if (line.is_ref || line.normalized.empty()) continue;

        std::string line_norm = utils::normalize(line.text);
        if (line_norm.empty()) continue;

        // Search for best matching substring in the full transcript
        // starting from current position
        double best_start_ms = 0;
        double best_end_ms = 0;
        float best_score = 0;

        std::string transcript_norm = utils::normalize(full_transcript);

        // Forward search: find best match in the transcript
        for (size_t pos = 0; pos + line_norm.size() <= transcript_norm.size(); pos += 1) {
            std::string candidate = transcript_norm.substr(pos, line_norm.size());
            // Expand candidate window
            for (int expand = 0; expand < 20; expand++) {
                size_t end_pos = pos + line_norm.size() + expand;
                if (end_pos > transcript_norm.size()) break;
                std::string expanded = transcript_norm.substr(pos, end_pos - pos);
                double sim = utils::similarity(line_norm, expanded);
                if (sim > best_score) {
                    best_score = sim;
                    // Find which segments this spans
                    int char_start = pos;
                    int char_end = end_pos;
                    // Map char positions to segment timestamps
                    int seg_start = 0;
                    int seg_end = segments.size() - 1;
                    for (int s = 0; s < (int)segments.size(); s++) {
                        if (seg_char_offsets[s] <= char_start) seg_start = s;
                    }
                    for (int s = seg_start; s < (int)segments.size(); s++) {
                        if (seg_char_offsets[s + 1] >= char_end) { seg_end = s; break; }
                    }
                    best_start_ms = segments[seg_start].t0_ms;
                    best_end_ms = segments[seg_end].t1_ms;
                }
                if (sim > 0.7) break; // good enough
            }
            if (best_score > 0.7) break; // found a good match
        }

        AlignedLine al;
        al.line_index = line.index;
        al.start_ms = best_start_ms;
        al.end_ms = best_end_ms;
        al.confidence = best_score;
        al.text = line.text;
        result.lines.push_back(al);
    }

    result.success = true;
    fprintf(stderr, "[whisper] Aligned %zu lines\n", result.lines.size());
    return result;
}

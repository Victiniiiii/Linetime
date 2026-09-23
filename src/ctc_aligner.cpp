#include "ctc_aligner.h"
#include "utils.h"
#include "json.hpp"
#include <onnxruntime_c_api.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_set>

using json = nlohmann::json;
static const OrtApi* ort = nullptr;

struct FrameStamp {
    int phoneme_id;
    int start_frame;
    int end_frame;
    int target_idx;
    float confidence = 0.0f;
};

class ViterbiDecoder {
public:
    static constexpr int BLANK_ID = 0;
    static constexpr float NEG_INF = -1e18f;

    struct ViterbiResult {
        std::vector<int> frame_phonemes;
        std::vector<int> frame_phoneme_idx;
    };

    static ViterbiResult viterbi_decode(
        const float* log_probs, int T, int C,
        const std::vector<int>& ctc_path,
        const std::vector<int>& ctc_path_idx,
        int band_width = 0,
        int expected_end_frames = -1)
    {
        int ctc_len = ctc_path.size();
        if (ctc_len == 0 || T == 0) return {{}, {}};

        bool use_band = (band_width > 0 && T > 1 && ctc_len > 1);
        float pace = use_band ? (float)(ctc_len - 1) / (T - 1) : 0.0f;

        std::vector<std::vector<float>> dp(T, std::vector<float>(ctc_len, NEG_INF));
        std::vector<std::vector<int>> backpointers(T, std::vector<int>(ctc_len, 0));

        dp[0][0] = log_probs[ctc_path[0]];
        if (ctc_len > 1)
            dp[0][1] = log_probs[ctc_path[1]];

        std::vector<bool> can_skip(ctc_len, false);
        for (int s = 2; s < ctc_len; s++) {
            if (ctc_path[s] != ctc_path[s - 2])
                can_skip[s] = true;
        }

        for (int t = 1; t < T; t++) {
            for (int s = 0; s < ctc_len; s++) {
                float emit = log_probs[t * C + ctc_path[s]];
                float best_score = dp[t - 1][s] + emit;
                int best_prev = s;

                if (s >= 1) {
                    float advance = dp[t - 1][s - 1] + emit;
                    if (advance > best_score) { best_score = advance; best_prev = s - 1; }
                }
                if (s >= 2 && can_skip[s]) {
                    float skip = dp[t - 1][s - 2] + emit;
                    if (skip > best_score) { best_score = skip; best_prev = s - 2; }
                }
                dp[t][s] = best_score;
                backpointers[t][s] = best_prev;
            }
            if (use_band) {
                float center = t * pace;
                int lo = std::max(0, (int)(center - band_width));
                int hi = std::min(ctc_len - 1, (int)(center + band_width));
                for (int s = 0; s < lo; s++) dp[t][s] = NEG_INF;
                for (int s = hi + 1; s < ctc_len; s++) dp[t][s] = NEG_INF;
            }
        }

        // Force final_state = ctc_len-1 at t = T-1 (production algorithm).
        // This forces the Viterbi path to span the full audio duration,
        // preventing the path from racing through all tokens early.
        int final_state = ctc_len - 1;
        int final_t = T - 1;

        // If dp[T-1][ctc_len-1] is unreachable (NEG_INF), find the nearest reachable time
        if (dp[T - 1][ctc_len - 1] <= NEG_INF / 2) {
            final_t = -1;
            float best = NEG_INF;
            for (int t = T - 1; t >= 0; t--) {
                if (dp[t][ctc_len - 1] > best) {
                    best = dp[t][ctc_len - 1];
                    final_t = t;
                }
            }
            if (final_t < 0) {
                // fallback: best state at T-1
                final_t = T - 1;
                final_state = 0;
                float best2 = NEG_INF;
                for (int s = 0; s < ctc_len; s++) {
                    if (dp[T - 1][s] > best2) {
                        best2 = dp[T - 1][s];
                        final_state = s;
                    }
                }
            }
        }

        std::vector<int> path_states(final_t + 1);
        path_states[final_t] = final_state;
        for (int t = final_t - 1; t >= 0; t--)
            path_states[t] = backpointers[t + 1][path_states[t + 1]];

        ViterbiResult result;
        result.frame_phonemes.resize(final_t + 1);
        result.frame_phoneme_idx.resize(final_t + 1);
        for (int t = 0; t <= final_t; t++) {
            result.frame_phonemes[t] = ctc_path[path_states[t]];
            result.frame_phoneme_idx[t] = ctc_path_idx[path_states[t]];
        }
        return result;
    }

    static std::vector<FrameStamp> assort_frames(
        const std::vector<int>& frame_phonemes,
        const std::vector<int>& frame_phoneme_idx,
        int start_frame_offset,
        int max_blanks = 10)
    {
        if (frame_phonemes.empty()) return {};

        std::vector<int> transitions;
        transitions.push_back(0);
        for (size_t i = 1; i < frame_phonemes.size(); i++) {
            if (frame_phonemes[i] != frame_phonemes[i - 1] ||
                frame_phoneme_idx[i] != frame_phoneme_idx[i - 1]) {
                transitions.push_back(i);
            }
        }

        std::vector<FrameStamp> stamps;
        for (size_t i = 0; i < transitions.size(); i++) {
            int start = transitions[i];
            int end = (i + 1 < transitions.size())
                ? transitions[i + 1]
                : (int)frame_phonemes.size();

            int phoneme = frame_phonemes[start];
            int idx = frame_phoneme_idx[start];

            if (idx == -1) {
                for (int j = start; j < end; j++) {
                    if (frame_phoneme_idx[j] != -1) { idx = frame_phoneme_idx[j]; break; }
                }
            }

            if (phoneme == BLANK_ID) {
                if ((end - start) > max_blanks) continue;
            }

            if (phoneme != BLANK_ID) {
                stamps.push_back({phoneme, start + start_frame_offset, end + start_frame_offset, idx, 0.0f});
            }
        }
        return stamps;
    }
};

struct Tokenizer {
    std::map<std::string, int> token_to_id;
    std::map<int, std::string> id_to_token;
    int vocab_size = 0;
    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        json j; f >> j;
        for (auto& [key, value] : j.items()) {
            int id = value.get<int>();
            token_to_id[key] = id; id_to_token[id] = key;
        }
        vocab_size = token_to_id.size();
        fprintf(stderr, "[ctc] Loaded tokenizer: %d tokens\n", vocab_size);
        return true;
    }
    std::vector<int> tokenize(const std::string& text) const {
        std::vector<int> ids;
        std::string normalized = utils::normalize(text);
        for (char c : normalized) {
            std::string s(1, c);
            auto it = token_to_id.find(s);
            if (it != token_to_id.end()) ids.push_back(it->second);
        }
        return ids;
    }
};

struct CTCAligner::Impl {
    OrtEnv* env = nullptr;
    OrtSession* session = nullptr;
    OrtSessionOptions* session_opts = nullptr;
    Tokenizer tokenizer;
    bool initialized = false;
    ~Impl() {
        if (session) ort->ReleaseSession(session);
        if (session_opts) ort->ReleaseSessionOptions(session_opts);
        if (env) ort->ReleaseEnv(env);
    }
};

CTCAligner::CTCAligner() : impl_(new Impl()) {}
CTCAligner::~CTCAligner() { delete impl_; }

static std::vector<float> boost_target_phonemes(
    const float* log_probs, int T, int C,
    const std::vector<int>& target_tokens, float boost_factor = 5.0f)
{
    std::vector<float> boosted(log_probs, log_probs + T * C);

    std::unordered_set<int> unique_targets(target_tokens.begin(), target_tokens.end());
    unique_targets.erase(ViterbiDecoder::BLANK_ID);

    for (int phoneme_idx : unique_targets) {
        if (phoneme_idx >= 0 && phoneme_idx < C) {
            for (int t = 0; t < T; t++)
                boosted[t * C + phoneme_idx] += boost_factor;
        }
    }

    for (int t = 0; t < T; t++) {
        float mx = *std::max_element(boosted.data() + t*C, boosted.data() + (t+1)*C);
        float sum = 0;
        for (int c = 0; c < C; c++) { boosted[t*C+c] = std::exp(boosted[t*C+c] - mx); sum += boosted[t*C+c]; }
        for (int c = 0; c < C; c++) { boosted[t*C+c] = std::log(boosted[t*C+c] / sum + 1e-10f); }
    }
    return boosted;
}

static bool try_append_cuda(const OrtApi* ort, OrtSessionOptions* opts) {
    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id = 0;
    cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
    cuda_opts.gpu_mem_limit = 8ULL * 1024 * 1024 * 1024;
    cuda_opts.arena_extend_strategy = 1;
    cuda_opts.do_copy_in_default_stream = 1;
    OrtStatus* s = ort->SessionOptionsAppendExecutionProvider_CUDA(opts, &cuda_opts);
    if (s) {
        ort->ReleaseStatus(s);
        return false;
    }
    return true;
}

bool CTCAligner::init(const std::string& onnx_model_path,
                       const std::string& tokenizer_path,
                       Provider provider) {
    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort) { fprintf(stderr, "[ctc] Failed to get ONNX Runtime API\n"); return false; }
    if (!impl_->tokenizer.load(tokenizer_path)) {
        fprintf(stderr, "[ctc] Failed to load tokenizer from %s\n", tokenizer_path.c_str());
        return false;
    }
    OrtStatus* s;
    s = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "linetime", &impl_->env);
    if (s) { ort->ReleaseStatus(s); return false; }
    s = ort->CreateSessionOptions(&impl_->session_opts);
    if (s) { ort->ReleaseStatus(s); return false; }
    ort->SetIntraOpNumThreads(impl_->session_opts, 4);

    // Append execution providers based on requested provider
    const char* provider_name = "CPU";
    if (provider == Provider::CUDA || provider == Provider::Auto) {
        if (try_append_cuda(ort, impl_->session_opts)) {
            provider_name = "CUDA";
        } else if (provider == Provider::CUDA) {
            fprintf(stderr, "[ctc] CUDA requested but not available\n");
        }
    }
    // CoreML is macOS-only, handled via build flags (ORT_COREML)
#ifdef ORT_COREML
    if (provider == Provider::CoreML || provider == Provider::Auto) {
        if (provider_name == nullptr || strcmp(provider_name, "CPU") == 0) {
            OrtStatus* cm = ort->SessionOptionsAppendExecutionProvider_CoreML(impl_->session_opts, 0);
            if (cm) {
                ort->ReleaseStatus(cm);
            } else {
                provider_name = "CoreML";
            }
        }
    }
#endif

    fprintf(stderr, "[ctc] Execution provider: %s\n", provider_name);

    #ifdef _WIN32
    std::wstring model_wpath(onnx_model_path.begin(), onnx_model_path.end());
    s = ort->CreateSession(impl_->env, model_wpath.c_str(), impl_->session_opts, &impl_->session);
#else
    s = ort->CreateSession(impl_->env, onnx_model_path.c_str(), impl_->session_opts, &impl_->session);
#endif
    if (s) {
        fprintf(stderr, "[ctc] Failed to load model: %s\n", ort->GetErrorMessage(s));
        ort->ReleaseStatus(s); return false;
    }
    size_t ni = 0, no = 0;
    ort->SessionGetInputCount(impl_->session, &ni);
    ort->SessionGetOutputCount(impl_->session, &no);
    fprintf(stderr, "[ctc] Model loaded: %zu inputs, %zu outputs\n", ni, no);
    impl_->initialized = true;
    return true;
}

CTCAlignerResult CTCAligner::align(const AudioBuffer& audio, const LyricsDocument& lyrics, float boost) {
    CTCAlignerResult result;
    if (!impl_->initialized) { result.error = "CTC aligner not initialized"; return result; }
    if (audio.n_samples == 0) { result.error = "Empty audio"; return result; }

    constexpr int SAMPLE_RATE = 16000;

    std::vector<float> all_raw;
    int total_T = 0;
    int C = 0;

    {
        constexpr int CHUNK_SECONDS = 60;
        int chunk_samples = CHUNK_SECONDS * SAMPLE_RATE;
        int total_samples = audio.n_samples;
        int num_chunks = (total_samples + chunk_samples - 1) / chunk_samples;

        for (int chunk = 0; chunk < num_chunks; chunk++) {
            int start = chunk * chunk_samples;
            int end = std::min(start + chunk_samples, total_samples);
            int chunk_len = end - start;

            fprintf(stderr, "[ctc] Processing chunk %d/%d (%.1f-%.1f sec)...\n",
                    chunk + 1, num_chunks, start / 16000.0, end / 16000.0);

            std::vector<int64_t> input_shape = {1, chunk_len};
            OrtMemoryInfo* mem_info = nullptr;
            ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
            OrtValue* in_tensor = nullptr;
            ort->CreateTensorWithDataAsOrtValue(
                mem_info, (void*)(audio.samples.data() + start), chunk_len * sizeof(float),
                input_shape.data(), 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_tensor);

            // attention_mask: ones for all frames (required by MMS FA export)
            std::vector<int64_t> attn(chunk_len, 1);
            OrtValue* attn_tensor = nullptr;
            ort->CreateTensorWithDataAsOrtValue(
                mem_info, attn.data(), chunk_len * sizeof(int64_t),
                input_shape.data(), 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &attn_tensor);

            OrtAllocator* allocator = nullptr;
            ort->GetAllocatorWithDefaultOptions(&allocator);
            char* in_name = nullptr; char* in_name1 = nullptr; char* out_name = nullptr;
            ort->SessionGetInputName(impl_->session, 0, allocator, &in_name);
            size_t ni_local = 0;
            ort->SessionGetInputCount(impl_->session, &ni_local);
            if (ni_local >= 2) ort->SessionGetInputName(impl_->session, 1, allocator, &in_name1);
            ort->SessionGetOutputName(impl_->session, 0, allocator, &out_name);

            // Order inputs to match model: prefer input_values then attention_mask
            const char* in_names[2];
            OrtValue* in_tensors[2];
            int n_inputs = 1;
            bool use_attn = (in_name1 != nullptr);
            if (use_attn) {
                // Heuristic: name containing "input_values"/"audio" is first
                auto is_primary = [](const char* n) {
                    if (!n) return false;
                    std::string s(n);
                    return s.find("input_values") != std::string::npos ||
                           s.find("audio") != std::string::npos ||
                           s.find("input") != std::string::npos;
                };
                if (is_primary(in_name) && !is_primary(in_name1)) {
                    in_names[0] = in_name; in_tensors[0] = in_tensor;
                    in_names[1] = in_name1; in_tensors[1] = attn_tensor;
                } else if (is_primary(in_name1) && !is_primary(in_name)) {
                    in_names[0] = in_name1; in_tensors[0] = in_tensor;
                    in_names[1] = in_name; in_tensors[1] = attn_tensor;
                } else {
                    // default order: input 0 = float audio, input 1 = mask
                    in_names[0] = in_name; in_tensors[0] = in_tensor;
                    in_names[1] = in_name1; in_tensors[1] = attn_tensor;
                }
                n_inputs = 2;
            } else {
                in_names[0] = in_name; in_tensors[0] = in_tensor;
            }
            const char* out_names[] = {out_name};

            OrtValue* out_tensor = nullptr;
            OrtStatus* status = ort->Run(impl_->session, nullptr,
                in_names, in_tensors, n_inputs, out_names, 1, &out_tensor);
            if (status) {
                result.error = std::string("ONNX run failed: ") + ort->GetErrorMessage(status);
                ort->ReleaseStatus(status);
                ort->ReleaseValue(in_tensor);
                if (attn_tensor) ort->ReleaseValue(attn_tensor);
                ort->ReleaseMemoryInfo(mem_info);
                allocator->Free(allocator, in_name);
                if (in_name1) allocator->Free(allocator, in_name1);
                allocator->Free(allocator, out_name);
                return result;
            }

            OrtTensorTypeAndShapeInfo* shape_info = nullptr;
            ort->GetTensorTypeAndShape(out_tensor, &shape_info);
            size_t nd = 0; ort->GetDimensionsCount(shape_info, &nd);
            std::vector<int64_t> out_shape(nd);
            ort->GetDimensions(shape_info, out_shape.data(), nd);
            ort->ReleaseTensorTypeAndShapeInfo(shape_info);
            int T = out_shape[1]; C = out_shape[2];

            float* raw_output = nullptr;
            ort->GetTensorMutableData(out_tensor, (void**)&raw_output);

            all_raw.insert(all_raw.end(), raw_output, raw_output + T * C);
            total_T += T;

            ort->ReleaseValue(out_tensor);
            ort->ReleaseValue(in_tensor);
            if (attn_tensor) ort->ReleaseValue(attn_tensor);
            ort->ReleaseMemoryInfo(mem_info);
            allocator->Free(allocator, in_name);
            if (in_name1) allocator->Free(allocator, in_name1);
            allocator->Free(allocator, out_name);
        }
    }

    for (int t = 0; t < total_T; t++) {
        all_raw[t * C + 28] = -1e9f;
    }

    fprintf(stderr, "[ctc] Total: %d frames (%.1f sec), vocab: %d\n", total_T, audio.duration_sec, C);

    const double frame_ms = 20.0;

    // --- Pre-tokenize all alignable lines ---
    struct PendingLine {
        int index;
        std::string text;
        std::vector<int> tokens;
        bool has_tokens;
        double start_ms, end_ms;
        float conf;
        int order;
    };
    std::vector<PendingLine> pending;
    int total_tokens = 0;
    for (const auto& line : lyrics.lines) {
        if (line.is_ref) continue;
        PendingLine p;
        p.index = line.index;
        p.text = line.text;
        p.tokens = impl_->tokenizer.tokenize(line.text);
        p.has_tokens = !p.tokens.empty();
        p.start_ms = p.end_ms = -1;
        p.conf = 0;
        p.order = (int)pending.size();
        if (p.has_tokens) total_tokens += (int)p.tokens.size();
        pending.push_back(std::move(p));
    }

    // --- Global single Viterbi over concatenated CTC path (production algorithm) ---
    // One path for all lines: blank-separated tokens at fixed stride, band = ctc_len/3,
    // force final state at end of audio. Per-line windows caused mid-song drift.
    constexpr int STRIDE = 4;
    std::vector<int> ctc_path;
    std::vector<int> ctc_path_idx;          // token serial or -1 for blank
    std::vector<int> token_to_pending;      // token serial -> pending index

    ctc_path.push_back(ViterbiDecoder::BLANK_ID);
    ctc_path_idx.push_back(-1);

    int token_serial = 0;
    for (size_t li = 0; li < pending.size(); li++) {
        if (!pending[li].has_tokens) continue;
        for (size_t j = 0; j < pending[li].tokens.size(); j++) {
            // gap of STRIDE-1 blanks before each token (except we already have pos 0 blank)
            while ((int)ctc_path.size() < 1 + token_serial * STRIDE) {
                ctc_path.push_back(ViterbiDecoder::BLANK_ID);
                ctc_path_idx.push_back(-1);
            }
            // ensure position exists
            int pos = 1 + token_serial * STRIDE;
            if (pos >= (int)ctc_path.size()) {
                ctc_path.resize(pos + 1, ViterbiDecoder::BLANK_ID);
                ctc_path_idx.resize(pos + 1, -1);
            }
            ctc_path[pos] = pending[li].tokens[j];
            ctc_path_idx[pos] = token_serial;
            token_to_pending.push_back((int)li);
            token_serial++;
        }
        // one extra blank between lines
        ctc_path.push_back(ViterbiDecoder::BLANK_ID);
        ctc_path_idx.push_back(-1);
    }

    int ctc_len = (int)ctc_path.size();
    int n_token_lines = 0;
    for (auto& p : pending) if (p.has_tokens) n_token_lines++;

    fprintf(stderr, "[ctc] CTC path: %d positions for %d lines\n", ctc_len, n_token_lines);
    fprintf(stderr, "[ctc] Non-blank boost: %.1f\n", boost);

    int aligned_n = 0;
    if (ctc_len <= 1 || total_T <= 1 || total_tokens == 0) {
        // fall through to interpolation-only below
        fprintf(stderr, "[ctc] Empty CTC path — interpolating all lines\n");
    } else {
        int band_width = ctc_len / 3;
        fprintf(stderr, "[ctc] Viterbi band_width=%d, ctc_len=%d, T=%d\n", band_width, ctc_len, total_T);

        auto boosted = boost_target_phonemes(all_raw.data(), total_T, C,
            token_to_pending.empty() ? std::vector<int>{} : [&]() {
                std::vector<int> all;
                all.reserve(total_tokens);
                for (auto& p : pending) if (p.has_tokens)
                    all.insert(all.end(), p.tokens.begin(), p.tokens.end());
                return all;
            }(), boost);

        // Force completion: final_state = ctc_len-1 at t near T-1 (matches production 100%).
        // expected_end_frames = T-1 with strong distance weight pulls final to end.
        int expected_end = total_T - 1;
        auto vr = ViterbiDecoder::viterbi_decode(boosted.data(), total_T, C,
            ctc_path, ctc_path_idx, band_width, expected_end);

        // Extract final state report (production-style)
        {
            // Count how many path positions were "visited" as non-blank max serial
            int max_serial = -1;
            for (int idx : vr.frame_phoneme_idx) if (idx > max_serial) max_serial = idx;
            int final_state = max_serial + 1; // positions completed
            if (final_state < 0) final_state = 0;
            if (final_state > ctc_len) final_state = ctc_len;
            fprintf(stderr, "[ctc] Viterbi final_state=%d/%d (%.1f%%)\n",
                    final_state > 0 ? final_state - 1 : 0, ctc_len,
                    100.0 * final_state / ctc_len);
        }

        // Per-line times from global path
        std::vector<double> start_ms(pending.size(), -1);
        std::vector<double> end_ms(pending.size(), -1);
        std::vector<double> conf_sum(pending.size(), 0.0);
        std::vector<int> conf_n(pending.size(), 0);

        int final_t = (int)vr.frame_phonemes.size();
        for (int t = 0; t < final_t; t++) {
            int serial = vr.frame_phoneme_idx[t];
            if (serial < 0) continue;
            if (vr.frame_phonemes[t] == ViterbiDecoder::BLANK_ID) continue;
            if (serial >= (int)token_to_pending.size()) continue;
            int li = token_to_pending[serial];
            double ms = t * frame_ms;
            if (start_ms[li] < 0) start_ms[li] = ms;
            end_ms[li] = ms + frame_ms;
            conf_sum[li] += std::exp(boosted[t * C + vr.frame_phonemes[t]]);
            conf_n[li]++;
        }

        for (size_t i = 0; i < pending.size(); i++) {
            if (!pending[i].has_tokens) continue;
            if (start_ms[i] < 0) {
                // no frames hit — leave for interpolation
                continue;
            }
            pending[i].start_ms = start_ms[i];
            pending[i].end_ms = end_ms[i];
            pending[i].conf = conf_n[i] > 0 ? (float)(conf_sum[i] / conf_n[i]) : 0.f;
            aligned_n++;
            fprintf(stderr, "[ctc] Line %zu: [%.1f - %.1f] conf=%.3f\n",
                    i + 1, pending[i].start_ms / 1000.0, pending[i].end_ms / 1000.0,
                    pending[i].conf);
        }
        fprintf(stderr, "[ctc] Aligned %d lines\n", aligned_n);
    }

    // --- Interpolate empty-token / missed lines between neighbors ---
    int onset_frame = 0;
    for (size_t i = 0; i < pending.size(); i++) {
        if (pending[i].start_ms >= 0) continue;
        // find prev and next tokenized
        double prev_s = 0, prev_e = 0;
        bool has_prev = false;
        for (int j = (int)i - 1; j >= 0; j--) {
            if (pending[j].has_tokens && pending[j].start_ms >= 0) {
                prev_s = pending[j].start_ms;
                prev_e = pending[j].end_ms;
                has_prev = true;
                break;
            }
        }
        double next_s = -1;
        for (size_t j = i + 1; j < pending.size(); j++) {
            if (pending[j].has_tokens && pending[j].start_ms >= 0) {
                next_s = pending[j].start_ms;
                break;
            }
        }
        if (has_prev && next_s >= 0) {
            // place between prev_e and next_s proportionally among consecutive empties
            int gap_count = 1;
            for (size_t j = i + 1; j < pending.size(); j++) {
                if (pending[j].has_tokens) break;
                gap_count++;
            }
            int rank = 0;
            for (size_t j = i; j > 0; j--) {
                if (pending[j - 1].has_tokens) break;
                rank++;
            }
            double span = next_s - prev_e;
            if (span < 0) span = 0;
            pending[i].start_ms = prev_e + span * (rank + 1) / (gap_count + 1);
            pending[i].end_ms = pending[i].start_ms + std::min(3000.0, span / (gap_count + 1) + 800);
        } else if (has_prev) {
            pending[i].start_ms = prev_e + 200;
            pending[i].end_ms = pending[i].start_ms + 2000;
        } else if (next_s >= 0) {
            pending[i].start_ms = std::max(0.0, next_s - 3000);
            pending[i].end_ms = next_s;
        } else {
            pending[i].start_ms = pending[i].end_ms = onset_frame * frame_ms;
        }
        pending[i].conf = 0;
    }

    for (auto& p : pending) {
        AlignedLine al;
        al.line_index = p.index;
        al.start_ms = p.start_ms < 0 ? 0 : p.start_ms;
        al.end_ms = p.end_ms < 0 ? al.start_ms : p.end_ms;
        al.confidence = p.conf;
        al.text = p.text;
        result.lines.push_back(al);
    }

    result.success = true;
    fprintf(stderr, "[ctc] Aligned %d lines (%zu total incl. interpolated)\n", aligned_n, result.lines.size());
    return result;
}

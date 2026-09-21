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
        int band_width = 0)
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

        int final_state = 0;
        float best_final = NEG_INF;
        for (int s = 0; s < ctc_len; s++) {
            if (dp[T - 1][s] > best_final) {
                best_final = dp[T - 1][s];
                final_state = s;
            }
        }

        std::vector<int> path_states(T);
        path_states[T - 1] = final_state;
        for (int t = T - 2; t >= 0; t--)
            path_states[t] = backpointers[t + 1][path_states[t + 1]];

        ViterbiResult result;
        result.frame_phonemes.resize(T);
        result.frame_phoneme_idx.resize(T);
        for (int t = 0; t < T; t++) {
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

    s = ort->CreateSession(impl_->env, onnx_model_path.c_str(), impl_->session_opts, &impl_->session);
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

CTCAlignerResult CTCAligner::align(const AudioBuffer& audio, const LyricsDocument& lyrics) {
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

            OrtAllocator* allocator = nullptr;
            ort->GetAllocatorWithDefaultOptions(&allocator);
            char* in_name = nullptr; char* out_name = nullptr;
            ort->SessionGetInputName(impl_->session, 0, allocator, &in_name);
            ort->SessionGetOutputName(impl_->session, 0, allocator, &out_name);
            const char* in_names[] = {in_name};
            const char* out_names[] = {out_name};

            OrtValue* out_tensor = nullptr;
            OrtStatus* status = ort->Run(impl_->session, nullptr,
                in_names, &in_tensor, 1, out_names, 1, &out_tensor);
            if (status) {
                result.error = std::string("ONNX run failed: ") + ort->GetErrorMessage(status);
                ort->ReleaseStatus(status); ort->ReleaseValue(in_tensor);
                ort->ReleaseMemoryInfo(mem_info);
                allocator->Free(allocator, in_name); allocator->Free(allocator, out_name);
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

            ort->ReleaseValue(out_tensor); ort->ReleaseValue(in_tensor);
            ort->ReleaseMemoryInfo(mem_info);
            allocator->Free(allocator, in_name); allocator->Free(allocator, out_name);
        }
    }

    for (int t = 0; t < total_T; t++) {
        all_raw[t * C + 28] = -1e9f;
    }

    fprintf(stderr, "[ctc] Total: %d frames (%.1f sec), vocab: %d\n", total_T, audio.duration_sec, C);

    double frame_ms = 20.0;
    int prev_end_frame = 0;
    int line_count = 0;

    for (const auto& line : lyrics.lines) {
        if (line.is_ref || line.normalized.empty()) continue;
        line_count++;

        std::vector<int> tokens = impl_->tokenizer.tokenize(line.text);
        if (tokens.empty()) {
            AlignedLine al; al.line_index = line.index;
            al.start_ms = al.end_ms = 0; al.confidence = 0; al.text = line.text;
            result.lines.push_back(al); continue;
        }

        int S = tokens.size();
        int stride = 4;
        if (stride * S + 1 > (int)(total_T * 0.9)) stride = 3;
        if (stride * S + 1 > (int)(total_T * 0.8)) stride = 2;
        int ctc_len = stride * S + 1;

        std::vector<int> ctc_path(ctc_len, ViterbiDecoder::BLANK_ID);
        std::vector<int> ctc_path_idx(ctc_len, -1);
        for (int j = 0; j < S; j++) {
            int pos = 1 + j * stride;
            if (pos < ctc_len) {
                ctc_path[pos] = tokens[j];
                ctc_path_idx[pos] = j;
            }
        }

        int start_frame = prev_end_frame;
        int search_T = total_T - start_frame;
        if (search_T <= 0) {
            AlignedLine al; al.line_index = line.index;
            al.start_ms = total_T * frame_ms; al.end_ms = total_T * frame_ms;
            al.confidence = 0; al.text = line.text;
            result.lines.push_back(al); continue;
        }

        auto boosted = boost_target_phonemes(all_raw.data() + start_frame * C, search_T, C, tokens, 20.0f);

        int band_width = std::max(ctc_len / 3, 20);
        auto vr = ViterbiDecoder::viterbi_decode(boosted.data(), search_T, C, ctc_path, ctc_path_idx, band_width);

        auto stamps = ViterbiDecoder::assort_frames(vr.frame_phonemes, vr.frame_phoneme_idx, start_frame);

        double s_ms = -1, e_ms = -1;
        float conf = 0;
        int matched = 0;
        for (auto& st : stamps) {
            if (s_ms < 0) s_ms = st.start_frame * frame_ms;
            e_ms = (st.end_frame) * frame_ms;
            for (int t = st.start_frame; t < st.end_frame && t < total_T; t++) {
                conf += std::exp(boosted[(t - start_frame) * C + st.phoneme_id]);
                matched++;
            }
        }
        conf = matched > 0 ? conf / matched : 0;
        if (s_ms < 0) { s_ms = start_frame * frame_ms; e_ms = (start_frame + 1) * frame_ms; }

        if (e_ms > s_ms + 100) {
            prev_end_frame = (int)(e_ms / frame_ms) + 1;
        }

        AlignedLine al; al.line_index = line.index;
        al.start_ms = s_ms; al.end_ms = e_ms; al.confidence = conf; al.text = line.text;
        result.lines.push_back(al);
    }

    result.success = true;
    fprintf(stderr, "[ctc] Aligned %d lines\n", line_count);
    return result;
}

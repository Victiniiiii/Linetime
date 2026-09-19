#include "ctc_aligner.h"
#include "utils.h"
#include "json.hpp"
#include <onnxruntime_c_api.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>

using json = nlohmann::json;
static const OrtApi* ort = nullptr;

class ViterbiDecoder {
public:
    static constexpr int BLANK_ID = 0;
    static constexpr float NEG_INF = -1e9f;

    static std::vector<std::pair<int, int>> viterbi_decode(
        const float* log_probs, int T, int C, const std::vector<int>& target)
    {
        int S = target.size();
        if (S == 0 || T == 0) return {};
        std::vector<int> ctc_path;
        ctc_path.push_back(BLANK_ID);
        for (int i = 0; i < S; i++) {
            ctc_path.push_back(target[i]);
            ctc_path.push_back(BLANK_ID);
        }
        int ctc_len = ctc_path.size();
        std::vector<float> dp(ctc_len, NEG_INF);
        std::vector<std::vector<short>> backpointers(T, std::vector<short>(ctc_len, -1));
        dp[0] = log_probs[ctc_path[0]];
        if (ctc_len > 1) dp[1] = log_probs[ctc_path[1]];
        float pace = (float)(ctc_len - 1) / (float)std::max(T - 1, 1);
        int band_width = std::max(ctc_len / 3, 20);
        for (int t = 1; t < T; t++) {
            std::vector<float> new_dp(ctc_len, NEG_INF);
            int s_min = std::max(0, (int)(pace * t - band_width));
            int s_max = std::min(ctc_len - 1, (int)(pace * t + band_width));
            for (int s = s_min; s <= s_max; s++) {
                float emission = log_probs[t * C + ctc_path[s]];
                float best_score = NEG_INF;
                short best_ptr = -1;
                if (dp[s] > best_score) { best_score = dp[s]; best_ptr = s; }
                if (s > 0 && dp[s-1] > best_score) { best_score = dp[s-1]; best_ptr = s - 1; }
                if (s > 1 && ctc_path[s] != ctc_path[s-2] && dp[s-2] > best_score) {
                    best_score = dp[s-2]; best_ptr = s - 2;
                }
                new_dp[s] = best_score + emission;
                backpointers[t][s] = best_ptr;
            }
            dp = std::move(new_dp);
        }
        int best_end = ctc_len - 1;
        float best_score = dp[best_end];
        for (int s = ctc_len - 2; s >= 0; s -= 2) {
            if (dp[s] > best_score) { best_score = dp[s]; best_end = s; }
        }
        std::vector<int> path(T);
        int s = best_end;
        path[T-1] = s;
        for (int t = T - 2; t >= 0; t--) {
            s = backpointers[t+1][s];
            if (s < 0) s = 0;
            path[t] = s;
        }
        std::vector<std::pair<int, int>> boundaries(S, {-1, -1});
        for (int t = 0; t < T; t++) {
            int ctc_state = path[t];
            for (int i = 0; i < S; i++) {
                int expected_pos = 1 + i * 2;
                if (ctc_state == expected_pos || ctc_state == expected_pos + 1) {
                    if (boundaries[i].first == -1) boundaries[i].first = t;
                    boundaries[i].second = t;
                }
            }
        }
        return boundaries;
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

bool CTCAligner::init(const std::string& onnx_model_path,
                       const std::string& tokenizer_path) {
    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort) { fprintf(stderr, "[ctc] Failed to get ONNX Runtime API\n"); return false; }
    if (!impl_->tokenizer.load(tokenizer_path)) {
        fprintf(stderr, "[ctc] Failed to load tokenizer from %s\n", tokenizer_path.c_str());
        return false;
    }
    OrtStatus* s;
    s = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "sounddetect", &impl_->env);
    if (s) { ort->ReleaseStatus(s); return false; }
    s = ort->CreateSessionOptions(&impl_->session_opts);
    if (s) { ort->ReleaseStatus(s); return false; }
    ort->SetIntraOpNumThreads(impl_->session_opts, 4);
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

    std::vector<int64_t> input_shape = {1, audio.n_samples};
    OrtMemoryInfo* mem_info = nullptr;
    ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
    OrtValue* in_tensor = nullptr;
    ort->CreateTensorWithDataAsOrtValue(
        mem_info, (void*)audio.samples.data(), audio.n_samples * sizeof(float),
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
    int T = out_shape[1], C = out_shape[2];
    fprintf(stderr, "[ctc] Frames: %d (%.1f sec), vocab: %d\n", T, audio.duration_sec, C);

    float* logits = nullptr;
    ort->GetTensorMutableData(out_tensor, (void**)&logits);

    // Log-softmax
    std::vector<float> lp(T * C);
    for (int t = 0; t < T; t++) {
        float mx = *std::max_element(logits + t*C, logits + (t+1)*C);
        float sum = 0;
        for (int c = 0; c < C; c++) { lp[t*C+c] = std::exp(logits[t*C+c] - mx); sum += lp[t*C+c]; }
        for (int c = 0; c < C; c++) { lp[t*C+c] = std::log(lp[t*C+c] / sum + 1e-10f); }
    }

    double frame_ms = 20.0;
    for (const auto& line : lyrics.lines) {
        if (line.is_ref || line.normalized.empty()) continue;
        std::vector<int> tokens = impl_->tokenizer.tokenize(line.text);
        if (tokens.empty()) {
            AlignedLine al; al.line_index = line.index;
            al.start_ms = al.end_ms = 0; al.confidence = 0; al.text = line.text;
            result.lines.push_back(al); continue;
        }
        auto bnd = ViterbiDecoder::viterbi_decode(lp.data(), T, C, tokens);
        double s_ms = 0, e_ms = 0; float conf = 0;
        if (!bnd.empty() && bnd[0].first >= 0) {
            s_ms = bnd[0].first * frame_ms;
            int le = bnd.back().second >= 0 ? bnd.back().second : bnd.back().first;
            e_ms = (le + 1) * frame_ms;
            float tp = 0; int cnt = 0;
            for (size_t i = 0; i < tokens.size() && i < bnd.size(); i++) {
                if (bnd[i].first >= 0 && bnd[i].second >= 0)
                    for (int t = bnd[i].first; t <= bnd[i].second; t++) {
                        tp += std::exp(lp[t*C + tokens[i]]); cnt++;
                    }
            }
            conf = cnt > 0 ? tp / cnt : 0;
        }
        AlignedLine al; al.line_index = line.index;
        al.start_ms = s_ms; al.end_ms = e_ms; al.confidence = conf; al.text = line.text;
        result.lines.push_back(al);
    }

    result.success = true;
    ort->ReleaseValue(out_tensor); ort->ReleaseValue(in_tensor);
    ort->ReleaseMemoryInfo(mem_info);
    allocator->Free(allocator, in_name); allocator->Free(allocator, out_name);
    fprintf(stderr, "[ctc] Aligned %zu lines\n", result.lines.size());
    return result;
}

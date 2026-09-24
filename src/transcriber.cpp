#include "transcriber.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <map>
#include <unistd.h>

namespace fs = std::filesystem;

// Simple JSON parser for our specific structure
struct JsonValue {
    enum Type { Null, Bool, Number, String, Object, Array } type = Null;
    bool b = false;
    double n = 0;
    std::string s;
    std::map<std::string, JsonValue> o;
    std::vector<JsonValue> a;

    JsonValue() = default;
    JsonValue(bool v) : type(Bool), b(v) {}
    JsonValue(double v) : type(Number), n(v) {}
    JsonValue(const std::string& v) : type(String), s(v) {}
    JsonValue(const char* v) : type(String), s(v) {}
    JsonValue(std::map<std::string, JsonValue>&& v) : type(Object), o(std::move(v)) {}
    JsonValue(std::vector<JsonValue>&& v) : type(Array), a(std::move(v)) {}

    const JsonValue& operator[](const std::string& key) const {
        static JsonValue null;
        auto it = o.find(key);
        return it != o.end() ? it->second : null;
    }
    const JsonValue& operator[](size_t idx) const {
        static JsonValue null;
        return idx < a.size() ? a[idx] : null;
    }
    bool is_null() const { return type == Null; }
    std::string as_string() const { return s; }
    double as_number() const { return n; }
    bool as_bool() const { return b; }
};

static JsonValue parse_json(const char*& p);

static void skip_ws(const char*& p) {
    while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
}

static std::string parse_string(const char*& p) {
    std::string result;
    if (*p != '"') return result;
    p++;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            switch (*p) {
                case '"': result += '"'; break;
                case '\\': result += '\\'; break;
                case '/': result += '/'; break;
                case 'b': result += '\b'; break;
                case 'f': result += '\f'; break;
                case 'n': result += '\n'; break;
                case 'r': result += '\r'; break;
                case 't': result += '\t'; break;
                case 'u': result += '?'; p += 4; break; // skip unicode
                default: result += *p; break;
            }
        } else {
            result += *p;
        }
        p++;
    }
    if (*p == '"') p++;
    return result;
}

static JsonValue parse_value(const char*& p) {
    skip_ws(p);
    if (!*p) return JsonValue();
    if (*p == '"') return JsonValue(parse_string(p));
    if (*p == '{') {
        p++;
        std::map<std::string, JsonValue> obj;
        skip_ws(p);
        if (*p == '}') { p++; return JsonValue(std::move(obj)); }
        while (*p) {
            std::string key = parse_string(p);
            skip_ws(p);
            if (*p != ':') break;
            p++;
            JsonValue val = parse_value(p);
            obj[key] = std::move(val);
            skip_ws(p);
            if (*p == '}') { p++; break; }
            if (*p != ',') break;
            p++;
            skip_ws(p);
        }
        return JsonValue(std::move(obj));
    }
    if (*p == '[') {
        p++;
        std::vector<JsonValue> arr;
        skip_ws(p);
        if (*p == ']') { p++; return JsonValue(std::move(arr)); }
        while (*p) {
            arr.push_back(parse_value(p));
            skip_ws(p);
            if (*p == ']') { p++; break; }
            if (*p != ',') break;
            p++;
            skip_ws(p);
        }
        return JsonValue(std::move(arr));
    }
    if (*p == 't' && strncmp(p, "true", 4) == 0) { p += 4; return JsonValue(true); }
    if (*p == 'f' && strncmp(p, "false", 5) == 0) { p += 5; return JsonValue(false); }
    if (*p == 'n' && strncmp(p, "null", 4) == 0) { p += 4; return JsonValue(); }
    bool neg = false;
    if (*p == '-') { neg = true; p++; }
    double val = 0;
    while (*p >= '0' && *p <= '9') { val = val * 10 + (*p - '0'); p++; }
    if (*p == '.') {
        p++;
        double mult = 0.1;
        while (*p >= '0' && *p <= '9') { val += (*p - '0') * mult; mult *= 0.1; p++; }
    }
    if (*p == 'e' || *p == 'E') {
        p++;
        bool eneg = false;
        if (*p == '-') { eneg = true; p++; }
        else if (*p == '+') { p++; }
        int exp = 0;
        while (*p >= '0' && *p <= '9') { exp = exp * 10 + (*p - '0'); p++; }
        double mult = 1;
        for (int i = 0; i < exp; i++) mult *= 10;
        if (eneg) val /= mult; else val *= mult;
    }
    return JsonValue(neg ? -val : val);
}

static JsonValue parse_json(const char*& p) {
    return parse_value(p);
}

static std::string read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return "";
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s;
    s.resize(sz);
    fread(s.data(), 1, sz, f);
    fclose(f);
    return s;
}

static long parse_time_ms(const std::string& ts) {
    // format: "HH:MM:SS,mmm" or "HH:MM:SS.mmm"
    int h = 0, m = 0, s = 0, ms = 0;
    if (sscanf(ts.c_str(), "%d:%d:%d,%d", &h, &m, &s, &ms) == 4) {
        return ((h * 60 + m) * 60 + s) * 1000 + ms;
    }
    if (sscanf(ts.c_str(), "%d:%d:%d.%d", &h, &m, &s, &ms) == 4) {
        return ((h * 60 + m) * 60 + s) * 1000 + ms;
    }
    return 0;
}

static WhisperWord token_to_word(const JsonValue& tok) {
    WhisperWord w;
    w.text = tok["text"].as_string();
    w.start_ms = tok["offsets"]["from"].as_number();
    w.end_ms = tok["offsets"]["to"].as_number();
    w.prob = (float)tok["p"].as_number();
    return w;
}

static bool is_word_boundary(const std::string& tok_text) {
    return !tok_text.empty() && (tok_text[0] == ' ' || tok_text == "[_BEG_]" || tok_text == "[_END_]" || tok_text == "[_PAD_]" || tok_text == "[_UNK_]" || tok_text == "[_MASK_]" || tok_text == "[_SOS_]" || tok_text == "[_EOS_]" || tok_text == "[_TT_]" || tok_text == "[_NO_TIMESTAMPS_]" || tok_text == "[_LANGUAGE_]" || tok_text == "[_TASK_]" || tok_text == "[_TRANSCRIBE_]" || tok_text == "[_TRANSLATE_]" || tok_text == "[_SOT_]" || tok_text == "[_EOT_]" || tok_text == "[_PAD_]" || tok_text == "[_SOT_PREV_]");
}

TranscriptionResult transcribe_audio(const std::string& audio_path,
                                     const std::string& model_path,
                                     const std::string& language,
                                     int threads) {
    TranscriptionResult result;

    // Build command
    std::string tmp_base = "/tmp/linetime_transcribe_" + std::to_string(getpid());
    std::string cmd = "LD_LIBRARY_PATH=dist/lib dist/bin/whisper-cli "
        "-m " + model_path + " "
        "-l " + language + " "
        "-f " + audio_path + " "
        "-ojf -of " + tmp_base + " "
        "-t " + std::to_string(threads) + " "
        "-np 2>&1";

    fprintf(stderr, "[transcriber] Running: %s\n", cmd.c_str());

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        result.error = "Failed to run whisper-cli";
        return result;
    }

    char buffer[4096];
    std::string stderr_output;
    while (fgets(buffer, sizeof(buffer), pipe)) {
        stderr_output += buffer;
    }
    int status = pclose(pipe);
    if (status != 0) {
        result.error = "whisper-cli failed (exit " + std::to_string(status) + "): " + stderr_output;
        return result;
    }

    std::string json_path = tmp_base + ".json";
    std::string json_content = read_file(json_path);
    if (json_content.empty()) {
        result.error = "Failed to read JSON output: " + json_path;
        return result;
    }

    // Parse JSON
    const char* p = json_content.c_str();
    JsonValue root = parse_json(p);
    if (root.is_null()) {
        result.error = "Failed to parse JSON";
        return result;
    }

    result.language = root["result"]["language"].as_string();

    const JsonValue& transcription = root["transcription"];
    if (transcription.type != JsonValue::Array) {
        result.error = "transcription is not an array";
        return result;
    }

    for (size_t si = 0; si < transcription.a.size(); si++) {
        const JsonValue& seg = transcription.a[si];
        WhisperSegment segment;
        segment.text = seg["text"].as_string();
        segment.start_ms = seg["offsets"]["from"].as_number();
        segment.end_ms = seg["offsets"]["to"].as_number();

        const JsonValue& tokens = seg["tokens"];
        if (tokens.type == JsonValue::Array) {
            // Merge subword tokens into words
            std::vector<WhisperWord> word_tokens;
            for (size_t ti = 0; ti < tokens.a.size(); ti++) {
                const JsonValue& tok = tokens.a[ti];
                std::string ttext = tok["text"].as_string();
                if (ttext == "[_BEG_]" || ttext == "[_END_]" || ttext == "[_PAD_]" || 
                    ttext == "[_UNK_]" || ttext == "[_MASK_]" || ttext == "[_SOS_]" || 
                    ttext == "[_EOS_]" || ttext == "[_TT_]" || ttext == "[_NO_TIMESTAMPS_]" ||
                    ttext == "[_LANGUAGE_]" || ttext == "[_TASK_]" || ttext == "[_TRANSCRIBE_]" ||
                    ttext == "[_TRANSLATE_]" || ttext == "[_SOT_]" || ttext == "[_EOT_]" ||
                    ttext == "[_SOT_PREV_]") {
                    continue;
                }
                WhisperWord w = token_to_word(tok);
                if (word_tokens.empty() || is_word_boundary(ttext)) {
                    // New word
                    if (!word_tokens.empty() && w.text[0] == ' ') {
                        w.text = w.text.substr(1);
                    }
                    word_tokens.push_back(w);
                } else {
                    // Subword continuation
                    if (!word_tokens.empty()) {
                        word_tokens.back().text += ttext;
                        word_tokens.back().end_ms = w.end_ms;
                        word_tokens.back().prob = std::min(word_tokens.back().prob, w.prob);
                    }
                }
            }
            segment.words = std::move(word_tokens);
        }
        result.segments.push_back(std::move(segment));
    }

    // Clean up temp files
    std::remove((tmp_base + ".json").c_str());
    std::remove((tmp_base + ".txt").c_str());
    std::remove((tmp_base + ".vtt").c_str());
    std::remove((tmp_base + ".srt").c_str());
    std::remove((tmp_base + ".tsv").c_str());
    std::remove((tmp_base + ".wts").c_str());
    std::remove((tmp_base + ".lrc").c_str());

    result.success = true;
    return result;
}
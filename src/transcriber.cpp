#include "transcriber.h"
#include "child_process.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <sys/stat.h>
#endif

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
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) return "";
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return "";
    }
    long size = ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return "";
    }
    std::string content(static_cast<size_t>(size), '\0');
    size_t read = 0;
    if (!content.empty())
        read = fread(content.data(), 1, content.size(), file);
    content.resize(read);
    fclose(file);
    return content;
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

// Whisper control/special tokens: [\_BEG_], [\_TT_1499], etc.
static bool is_special_token(const std::string& t) {
    return t.size() >= 2 && t[0] == '[' && t[1] == '_';
}

// Folded to lower case with punctuation and spacing removed, so a repeat is
// recognised regardless of how whisper chose to capitalise or space it.
static std::string repetition_key(const std::string& text) {
    std::string key;
    key.reserve(text.size());
    for (char c : text) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= 'A' && uc <= 'Z') c = static_cast<char>(uc - 'A' + 'a');
        if ((uc >= 'a' && uc <= 'z') || (uc >= '0' && uc <= '9')) key.push_back(c);
    }
    return key;
}

// whisper locks onto a phrase and repeats it verbatim when a passage has no
// intelligible speech. It exits cleanly and reports high token confidence, so it
// cannot be filtered afterwards by score alone. A run of identical adjacent
// segments is therefore dropped, keeping the first, which is the only one that
// carried real timing information.
static void drop_repeated_segments(std::vector<WhisperSegment>& segments) {
    std::vector<WhisperSegment> kept;
    kept.reserve(segments.size());
    size_t dropped = 0;
    for (WhisperSegment& segment : segments) {
        const std::string key = repetition_key(segment.text);
        if (!key.empty() && !kept.empty() && repetition_key(kept.back().text) == key) {
            dropped++;
            continue;
        }
        kept.push_back(std::move(segment));
    }
    if (dropped > 0) {
        fprintf(stderr, "  Dropped %zu repeated segment(s) from the transcript\n", dropped);
    }
    // Always written back: the loop above moves out of `segments`, so leaving it
    // untouched would replace every segment with an empty one.
    segments = std::move(kept);
}

// Parse a whisper-cli -ojf JSON file into a TranscriptionResult
TranscriptionResult load_transcription_json(const std::string& json_path) {
    TranscriptionResult result;

    std::string json_content = read_file(json_path);
    if (json_content.empty()) {
        result.error = "Failed to read JSON: " + json_path;
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
    if (root["systeminfo"].type == JsonValue::String) {
        result.systeminfo = root["systeminfo"].as_string();
    }

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
                if (is_special_token(ttext)) continue;
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

    drop_repeated_segments(result.segments);

    result.success = true;
    return result;
}

class PrivateTempDirectory {
public:
    bool create(std::string& error) {
        std::error_code code;
        fs::path base = fs::temp_directory_path(code);
        if (code) {
            error = "failed to locate temporary directory: " + code.message();
            return false;
        }

        for (int attempt = 0; attempt < 100; ++attempt) {
            long long tick = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            unsigned long long sequence = temp_sequence.fetch_add(1);
            std::string name = "linetime-" +
                std::to_string(process::current_process_id()) + "-" +
                std::to_string(tick) + "-" + std::to_string(sequence);
            fs::path candidate = base / name;
            code.clear();
#ifdef _WIN32
            bool created = fs::create_directory(candidate, code);
#else
            bool created = ::mkdir(candidate.c_str(), S_IRUSR | S_IWUSR | S_IXUSR) == 0;
            if (!created && errno != EEXIST)
                code = std::error_code(errno, std::generic_category());
#endif
            if (created && code) {
                std::error_code ignored;
                fs::remove_all(candidate, ignored);
                code.clear();
                continue;
            }
            if (!created || code) continue;

#ifndef _WIN32
            code.clear();
            fs::permissions(candidate,
                            fs::perms::owner_read | fs::perms::owner_write |
                                fs::perms::owner_exec,
                            fs::perm_options::replace, code);
            if (code) {
                std::error_code ignored;
                fs::remove_all(candidate, ignored);
                error = "failed to secure temporary directory: " + code.message();
                return false;
            }
#endif
            path_ = candidate;
            return true;
        }
        error = "failed to create a unique temporary directory";
        return false;
    }

    ~PrivateTempDirectory() {
        if (path_.empty()) return;
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
    static inline std::atomic<unsigned long long> temp_sequence{0};
};

static bool write_pcm16_wav(const fs::path& path, const AudioBuffer& audio,
                            std::string& error) {
    if (audio.sample_rate != 16000) {
        error = "audio buffer is not 16 kHz";
        return false;
    }
    if (audio.n_samples < 0 ||
        static_cast<size_t>(audio.n_samples) > audio.samples.size()) {
        error = "audio buffer sample count is invalid";
        return false;
    }

    size_t sample_count = audio.n_samples == 0
                              ? audio.samples.size()
                              : static_cast<size_t>(audio.n_samples);
    if (sample_count > (std::numeric_limits<uint32_t>::max() - 36) / 2) {
        error = "audio buffer is too large for a WAV file";
        return false;
    }

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        error = "failed to create temporary WAV file";
        return false;
    }

    auto write_u16 = [&output](uint16_t value) {
        char bytes[2] = {
            static_cast<char>(value & 0xff),
            static_cast<char>((value >> 8) & 0xff)
        };
        output.write(bytes, sizeof(bytes));
    };
    auto write_u32 = [&output](uint32_t value) {
        char bytes[4] = {
            static_cast<char>(value & 0xff),
            static_cast<char>((value >> 8) & 0xff),
            static_cast<char>((value >> 16) & 0xff),
            static_cast<char>((value >> 24) & 0xff)
        };
        output.write(bytes, sizeof(bytes));
    };

    uint32_t data_size = static_cast<uint32_t>(sample_count * 2);
    output.write("RIFF", 4);
    write_u32(36 + data_size);
    output.write("WAVE", 4);
    output.write("fmt ", 4);
    write_u32(16);
    write_u16(1);
    write_u16(1);
    write_u32(16000);
    write_u32(16000 * 2);
    write_u16(2);
    write_u16(16);
    output.write("data", 4);
    write_u32(data_size);

    for (size_t i = 0; i < sample_count; ++i) {
        float value = audio.samples[i];
        if (!std::isfinite(value)) value = 0.0f;
        value = std::max(-1.0f, std::min(1.0f, value));
        int pcm = value < 0.0f
                      ? static_cast<int>(std::lround(value * 32768.0f))
                      : static_cast<int>(std::lround(value * 32767.0f));
        pcm = std::max(-32768, std::min(32767, pcm));
        write_u16(static_cast<uint16_t>(static_cast<int16_t>(pcm)));
    }
    output.close();
    if (!output) {
        error = "failed to write temporary WAV file";
        return false;
    }
    return true;
}

static void cache_transcript(const fs::path& source, const std::string& cache_name) {
    const char* configured = std::getenv("LTC_CACHE_DIR");
    if (!configured || !*configured) return;

    std::error_code code;
    fs::path directory(configured);
    fs::create_directories(directory, code);
    if (code) {
        fprintf(stderr, "[transcriber] Cache directory unavailable: %s\n",
                code.message().c_str());
        return;
    }

    std::string name = cache_name;
    if (name.empty()) {
        uint64_t hash = 1469598103934665603ULL;
        std::ifstream input(source, std::ios::binary);
        char buffer[8192];
        while (input) {
            input.read(buffer, sizeof(buffer));
            std::streamsize count = input.gcount();
            for (std::streamsize i = 0; i < count; ++i) {
                hash ^= static_cast<unsigned char>(buffer[i]);
                hash *= 1099511628211ULL;
            }
        }
        name = "audio_" + std::to_string(hash);
    } else {
        name = fs::path(name).filename().string();
        if (name.size() >= 5 && name.compare(name.size() - 5, 5, ".json") == 0)
            name.resize(name.size() - 5);
    }
    if (name.empty()) name = "transcript";

    fs::path destination = directory / (name + ".json");
    code.clear();
    fs::copy_file(source, destination,
                  fs::copy_options::overwrite_existing, code);
    if (code) {
        fprintf(stderr, "[transcriber] Cache copy failed: %s\n",
                code.message().c_str());
        return;
    }
    fprintf(stderr, "[transcriber] Cached transcript -> %s\n",
            destination.string().c_str());
}

static TranscriptionResult transcribe_single_language(const AudioBuffer& audio,
                                                      const std::string& model_path,
                                                      const std::string& whisper_cli_path,
                                                      const std::string& language,
                                                      Provider provider,
                                                      int threads,
                                                      const std::vector<std::string>& environment_additions,
                                                      const std::string& cache_name) {
    TranscriptionResult result;
    PrivateTempDirectory temporary;
    try {
        std::string error;
        if (!temporary.create(error)) {
            result.error = error;
            return result;
        }
        if (audio.samples.empty() || audio.n_samples < 0) {
            result.error = "audio buffer is empty";
            return result;
        }
        if (model_path.empty()) {
            result.error = "Whisper model path is empty";
            return result;
        }
        if (whisper_cli_path.empty()) {
            result.error = "whisper-cli path is empty";
            return result;
        }
        if (language.empty()) {
            result.error = "Whisper language is empty";
            return result;
        }

        fs::path wav_path = temporary.path() / "audio.wav";
        if (!write_pcm16_wav(wav_path, audio, error)) {
            result.error = error;
            return result;
        }
        fs::path output_prefix = temporary.path() / "transcript";
        fs::path json_path = temporary.path() / "transcript.json";

        int thread_count = threads > 0 ? threads : 16;
        std::vector<std::string> arguments = {
            whisper_cli_path,
            "-m", model_path,
            "-l", language,
            "-f", wav_path.string(),
            "-ojf",
            "-of", output_prefix.string(),
            "-t", std::to_string(thread_count),
            "-ml", "60",
            // Context conditioning feeds the previous segment back in as a prompt.
            // On instrumental passages that locks the decoder into a single phrase
            // and it then repeats for the whole track, so it is disabled.
            "-mc", "0",
            // Drops [MUSIC], [BLINK] and similar non-speech artefacts.
            "-sns"
            // Note: -np is deliberately not passed. It suppresses whisper's own
            // stderr, which includes the ggml_cuda_init line that is the only
            // reliable proof the run used the GPU. Passing it made every GPU run
            // look unverified, because the confirmation it depends on is exactly
            // what -np hides. progress_output() below drops the per-segment lines
            // from stderr instead, so the user-facing output is unchanged.
        };
        if (provider == Provider::CPU)
            arguments.push_back("-ng");

          fprintf(stderr, "[transcriber] Running: %s\n", whisper_cli_path.c_str());
          process::Result execution = process::run(arguments, environment_additions);
          // ggml_cuda_init reports the devices it found on stderr, and only when the
          // CUDA backend actually initialised. This is the only signal that
          // distinguishes a real GPU run from a CPU build that merely mentions CUDA
          // in its banner, so it is captured from the child's own output rather than
          // guessed from the flags passed to it.
          if (execution.stderr_data.find("ggml_cuda_init: found") != std::string::npos) {
              result.gpu_confirmed = true;
          }
          // whisper narrates every 30s window it decodes. Those lines are progress
          // noise in the middle of our own stage output, so they are dropped here
          // rather than by asking whisper to stay quiet, because the one line we
          // need from it is on the same stream.
          {
              size_t at = 0;
              const std::string& data = execution.stderr_data;
              while (at < data.size()) {
                  size_t end = data.find('\n', at);
                  if (end == std::string::npos) end = data.size();
                  const std::string line = data.substr(at, end - at);
                  if (line.rfind("whisper_print_segment_callback", 0) != 0 &&
                      line.rfind("[print]", 0) != 0 &&
                      line.find("whisper_full_with_state: segment") == std::string::npos &&
                      !line.empty()) {
                      fputs(line.c_str(), stderr);
                      fputc('\n', stderr);
                  }
                  at = end + 1;
              }
          }
          if (!execution.started) {
              result.error = "failed to run whisper-cli: " + execution.error;
              return result;
          }
        if (execution.exit_code != 0 || !execution.error.empty()) {
            result.error = "whisper-cli failed (exit " +
                           std::to_string(execution.exit_code) + "): " +
                           execution.stderr_data;
            if (!execution.error.empty())
                result.error += "; " + execution.error;
            return result;
        }

          TranscriptionResult parsed = load_transcription_json(json_path.string());
          if (!parsed.success) return parsed;

          // A GPU run that quietly used the CPU is worse than a failed one: it
          // looks like it worked and takes minutes instead of seconds.
          //
          // Checking systeminfo for the word "CUDA" is not enough, and was tried
          // first: a CPU-only build still prints the CUDA line of that template,
          // listing which architectures it would support, so the string is present
          // either way and the check passed a build with no CUDA in it at all.
          //
          // What differs is what whisper actually loaded, so that is what is asked:
          // ggml_cuda_init prints "found N CUDA devices" only when the CUDA backend
          // initialises against a real device, and use_gpu says whether it was
          // asked to. A build without CUDA prints neither.
          if (provider == Provider::CUDA) {
                if (result.gpu_confirmed) {
                  fprintf(stderr, "[transcriber] GPU confirmed: %s\n",
                          parsed.systeminfo.c_str());
              } else {
                  result.error =
                      "whisper-cli ran without CUDA, so this was not a GPU run. The "
                      "whisper-cli in use has no CUDA build, or no accelerator was "
                      "found. Refusing to return a CPU result for a GPU request.";
                  return result;
              }
          }
        try {
            cache_transcript(json_path, cache_name);
        } catch (const std::exception& exception) {
            fprintf(stderr, "[transcriber] Cache failed: %s\n", exception.what());
        } catch (...) {
            fprintf(stderr, "[transcriber] Cache failed with an unknown exception\n");
        }
        return parsed;
    } catch (const std::exception& exception) {
        result.error = std::string("transcription failed: ") + exception.what();
    } catch (...) {
        result.error = "transcription failed with an unknown exception";
    }
    return result;
}

// --- multi-language merge -----------------------------------------------------
//
// A song is not one language. Sredinom is a Bosnian body with an English chorus,
// and a single value gets half of it wrong either way: under "auto" whisper hears
// the chorus and misreads the verses, under "bs" it reads the verses and mishears
// the chorus. Running it twice and keeping the better transcript for each stretch
// is the only way to hold both.
//
// The comparison is per segment, not per track. Comparing whole transcripts would
// let the majority language veto the minority one, which is precisely the case
// that needs fixing.

static double mean_word_probability(const WhisperSegment& segment) {
    if (segment.words.empty()) return 0.0;
    double total = 0.0;
    for (const auto& w : segment.words) total += w.prob;
    return total / (double)segment.words.size();
}

// Confidence, with a small reward for covering more words so a short confident
// guess cannot displace a longer accurate one.
static double segment_score(const WhisperSegment& segment) {
    return mean_word_probability(segment) + 0.05 * std::log1p((double)segment.words.size());
}

// Mean probability of whatever the other transcript put over the same audio. A
// stretch spoken by neither language returns 0 rather than counting as agreement.
static double coverage_score(const TranscriptionResult& other,
                             double start_ms, double end_ms) {
    double total = 0.0;
    int n = 0;
    for (const auto& seg : other.segments) {
        if (!(start_ms < seg.end_ms && seg.start_ms < end_ms)) continue;
        total += mean_word_probability(seg);
        n++;
    }
    return n > 0 ? total / (double)n : 0.0;
}

TranscriptionResult transcribe_audio(const AudioBuffer& audio,
                                     const std::string& model_path,
                                     const std::string& whisper_cli_path,
                                     const std::string& language,
                                     Provider provider,
                                     int threads,
                                     const std::vector<std::string>& environment_additions,
                                     const std::string& cache_name) {
    // One language is the common case and is not worth a second run.
    size_t split = language.find('+');
    if (split == std::string::npos) {
        return transcribe_single_language(audio, model_path, whisper_cli_path, language,
                                          provider, threads, environment_additions, cache_name);
    }

    std::string primary = language.substr(0, split);
    std::string secondary = language.substr(split + 1);
    fprintf(stderr, "[transcriber] Multi-language %s + %s: transcribing twice and merging\n",
            primary.c_str(), secondary.c_str());

    TranscriptionResult a = transcribe_single_language(audio, model_path, whisper_cli_path,
                                                      primary, provider, threads,
                                                      environment_additions, cache_name);
    TranscriptionResult b = transcribe_single_language(audio, model_path, whisper_cli_path,
                                                      secondary, provider, threads,
                                                      environment_additions, cache_name);
    if (!a.success) return a;
    if (!b.success) return b;

    // The primary transcript is the spine. Its segments keep their place and
    // duration; only the words inside them can change, so merging cannot alter how
    // many lines there are or where they sit.
    TranscriptionResult result;
    result.success = true;
    result.language = primary + "+" + secondary;
    result.systeminfo = a.systeminfo;
    result.gpu_confirmed = a.gpu_confirmed;
    int replaced = 0;
    for (const auto& seg : a.segments) {
        WhisperSegment best = seg;
        if (coverage_score(b, seg.start_ms, seg.end_ms) > segment_score(seg) + 0.05) {
            std::vector<WhisperWord> words;
            std::string text;
            for (const auto& other : b.segments) {
                if (!(seg.start_ms < other.end_ms && other.start_ms < seg.end_ms)) continue;
                for (const auto& w : other.words) words.push_back(w);
                if (!text.empty()) text += " ";
                text += other.text;
            }
            if (!words.empty() && !text.empty()) {
                best.text = text;
                best.words = words;
                replaced++;
            }
        }
        result.segments.push_back(best);
    }
    fprintf(stderr, "[transcriber] Merge: %d of %zu segments took %s words\n",
            replaced, result.segments.size(), secondary.c_str());
    return result;
}

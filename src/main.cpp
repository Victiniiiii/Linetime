#include "audio.h"
#include "lyrics.h"
#include "ctc_aligner.h"
#include "lrc_writer.h"
#include "transcriber.h"
#include "reconcile.h"
#include "child_process.h"
#include "utils.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

// Non-lyrical vocals. Whisper has to emit something for every 30s window, so a
// hum becomes "ha ha ha" and a silent tail becomes a credit line. The distinction
// that matters is whether any *word* was recognised: a hum yields interjections at
// most, while a real lyric yields words with word-level timings. This is checked on
// the word list rather than the text, because "ha ha ha" is three words in the
// transcript and would otherwise look like content.
static bool is_interjection_word(const std::string& w) {
    static const char* const kInterjections[] = {
        "ha", "haha", "hahaha", "ah", "ahh", "eh", "hm", "hmm", "hmmm",
        "la", "lalala", "nana", "na", "nanana", "da", "dada", "ta", "tata",
        "oh", "ooh", "ooh ooh", "woo", "woohoo", "hey", "he", "heh", "ho",
        "mmm", "mm", "mhm", "huh", "yay", "yeah", "yay yeah",
        nullptr
    };
    std::string n = utils::normalize(w);
    if (n.empty()) return false;
    for (int i = 0; kInterjections[i]; i++) {
        if (n == kInterjections[i]) return true;
    }
    return false;
}

// True when nothing recognisable was transcribed for this segment. Reported to the
// caller as [humming] rather than dropped, so the timing of a vocal passage that
// carries no words survives into the LRC instead of leaving a silent gap.
static bool segment_has_no_words(const WhisperSegment& seg) {
    // A credit line or a real lyric both produce words here. Only a hum, an
    // instrumental passage or a silence produces none, and those are the cases the
    // placeholder exists for.
    for (const auto& w : seg.words) {
        if (!is_interjection_word(w.text)) return false;
    }
    return true;
}

// The placeholder written for such a segment. Bracketed so it is obviously not a
// lyric and is easy to find and edit by hand afterwards.
static const char* kNonLyricPlaceholder = "[humming]";

static std::vector<AlignedLine> segments_to_lines(const TranscriptionResult& trans) {
    std::vector<AlignedLine> out;
    for (const auto& seg : trans.segments) {
        std::string text = seg.text;
        size_t b = text.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        size_t e = text.find_last_not_of(" \t\r\n");
        text = text.substr(b, e - b + 1);
        for (size_t i = 0; i < text.size(); i++) {
            if (text[i] == '\n' || text[i] == '\r' || text[i] == '\t') text[i] = ' ';
        }
        bool in_space = false;
        std::string clean;
        clean.reserve(text.size());
        for (char c : text) {
            if (c == ' ') {
                if (!in_space) {
                    clean += ' ';
                    in_space = true;
                }
            } else {
                clean += c;
                in_space = false;
            }
        }
        if (clean.empty()) continue;

        // A segment with no recognisable words keeps its slot in the timeline as a
        // placeholder, and carries no align_text so CTC is not asked to force a
        // word onto a hum it cannot place.
        const bool non_lyric = segment_has_no_words(seg);

        float conf_sum = 0.0f;
        int conf_n = 0;
        for (const auto& w : seg.words) {
            conf_sum += w.prob;
            conf_n++;
        }

        AlignedLine al;
        al.line_index = (int)out.size();
        al.start_ms = seg.start_ms;
        al.end_ms = seg.end_ms;
        al.confidence = conf_n > 0 ? conf_sum / conf_n : 0.0f;
        al.text = non_lyric ? kNonLyricPlaceholder : clean;
        al.align_text = non_lyric ? "" : clean;
        out.push_back(std::move(al));
    }
    return out;
}

static int refine_with_ctc(std::vector<AlignedLine>& lines,
                           const AudioBuffer& audio,
                           float boost,
                           const std::string& model_a_path,
                           const std::string& tokenizer_path,
                           Provider provider,
                           int lead_ms) {
    LyricsDocument refined;
    std::vector<size_t> keep_idx;
    for (size_t i = 0; i < lines.size(); i++) {
        if (lines[i].text.empty()) continue;
        // A non-lyric placeholder has no word to align, so it is skipped here and
        // keeps the whisper timestamp it arrived with. Falling back to `text` for an
        // empty align_text would hand the literal string "[humming]" to CTC to force
        // onto audio, which is exactly the work this placeholder avoids.
        std::string align_text = lines[i].align_text.empty() ? lines[i].text : lines[i].align_text;
        if (align_text == kNonLyricPlaceholder) continue;
        LyricLine ll;
        ll.index = (int)refined.lines.size();
        ll.text = align_text;
        ll.normalized = ll.text;
        ll.is_ref = false;
        ll.is_expanded = false;
        refined.lines.push_back(ll);
        keep_idx.push_back(i);
    }
    refined.total_lines = (int)refined.lines.size();
    if (refined.lines.empty()) return 0;

    CTCAligner ctc;
    if (!ctc.init(model_a_path, tokenizer_path, provider)) {
        if (ctc.provider_unavailable()) return 1;
        fprintf(stderr, "  CTC refinement skipped (model not found; pass --model-a)\n");
        return 0;
    }
    CTCAlignerResult cta = ctc.align(audio, refined, boost);
    if (!cta.success || cta.lines.size() != refined.lines.size()) {
        fprintf(stderr, "  CTC refinement skipped (align failed)\n");
        return 0;
    }
    int nref = 0;
    for (size_t j = 0; j < keep_idx.size(); j++) {
        auto& line = lines[keep_idx[j]];
        const auto& cl = cta.lines[j];
        if (cl.confidence > 0.10f && cl.start_ms >= 0) {
            line.start_ms = std::max(0LL, (long long)cl.start_ms - lead_ms);
            line.end_ms = std::max(0LL, (long long)cl.end_ms - lead_ms);
            line.confidence = std::max(line.confidence, cl.confidence);
            nref++;
        }
    }
    return nref;
}

static bool parse_int_value(const std::string& text, int& value) {
    if (text.empty()) return false;
    long long parsed = 0;
    const char* begin = text.data();
    const char* end = begin + text.size();
    auto result = std::from_chars(begin, end, parsed, 10);
    if (result.ec != std::errc() || result.ptr != end ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max())
        return false;
    value = static_cast<int>(parsed);
    return true;
}

static bool parse_float_value(const std::string& text, float& value) {
    if (text.empty() ||
        std::isspace(static_cast<unsigned char>(text.front())) ||
        std::isspace(static_cast<unsigned char>(text.back())))
        return false;
    errno = 0;
    char* end = nullptr;
    double parsed = std::strtod(text.c_str(), &end);
    if (errno == ERANGE || end != text.c_str() + text.size() ||
        !std::isfinite(parsed) || parsed < -std::numeric_limits<float>::max() ||
        parsed > std::numeric_limits<float>::max())
        return false;
    value = static_cast<float>(parsed);
    return std::isfinite(value);
}

static bool take_value(int argc, char** argv, int& index,
                       const char* option, std::string& value) {
    if (index + 1 >= argc) {
        fprintf(stderr, "Error: %s requires a value\n", option);
        return false;
    }
    value = argv[++index];
    if (value.empty() || value[0] == '-') {
        fprintf(stderr, "Error: %s requires a value\n", option);
        return false;
    }
    return true;
}

static bool take_numeric_value(int argc, char** argv, int& index,
                               const char* option, std::string& value) {
    if (index + 1 >= argc || argv[index + 1][0] == '\0') {
        fprintf(stderr, "Error: %s requires a value\n", option);
        return false;
    }
    value = argv[++index];
    return !value.empty();
}

static bool existing_file(const fs::path& path) {
    std::error_code error;
    return fs::is_regular_file(path, error) && !error;
}

static std::string first_existing(const fs::path& base,
                                  const std::vector<fs::path>& relative_paths,
                                  const std::string& fallback) {
    for (const fs::path& relative : relative_paths) {
        fs::path candidate = base / relative;
        if (existing_file(candidate)) return candidate.string();
    }
    return fallback;
}

static std::string runtime_file(const fs::path& base, const char* relative) {
    return (base / relative).string();
}

static std::string prepend_runtime_path(const std::vector<fs::path>& directories,
                                        const char* variable) {
    std::string value;
#ifdef _WIN32
    const char separator = ';';
#else
    const char separator = ':';
#endif
    const char* current = std::getenv(variable);
    if (current) value = current;
    for (auto it = directories.rbegin(); it != directories.rend(); ++it) {
        std::string directory = it->string();
        if (directory.empty()) continue;
        std::string combined = directory;
        if (!value.empty()) combined += separator + value;
        value = std::move(combined);
    }
    return value;
}

static std::vector<std::string> runtime_environment(const fs::path& base,
                                                    const std::string& whisper_cli_path) {
    std::vector<fs::path> directories;
    directories.push_back(base / "lib");
    fs::path whisper_path(whisper_cli_path);
    if (whisper_path.has_parent_path()) {
        fs::path parent = whisper_path.parent_path();
        directories.push_back(parent / "lib");
        if (parent.filename() == "bin")
            directories.push_back(parent.parent_path() / "lib");
    }

#ifdef _WIN32
    const char* variable = "PATH";
#else
#ifdef __APPLE__
    const char* variable = "DYLD_LIBRARY_PATH";
#else
    const char* variable = "LD_LIBRARY_PATH";
#endif
#endif
    std::string path_value = prepend_runtime_path(directories, variable);
    if (path_value.empty()) return {};
    return {std::string(variable) + "=" + path_value};
}

void print_usage() {
    fprintf(stderr,
        "linetime v1.3 - Lyric-Audio Timestamp Aligner\n\n"
        "Usage: linetime <audio_file> [lyrics_file] [options]\n"
        "       cat lyrics.txt | linetime <audio_file> - [options]\n"
        "       linetime <audio_file> [options]     (no lyrics -> method b)\n\n"
        "Options:\n"
        "  -o, --output <path>      Output LRC file (default: <audio>.lrc)\n"
        "  --ffmpeg <path>          Path to ffmpeg binary\n"
        "  --model-a <path>         MMS_FA ONNX model\n"
        "  --model-c <path>         Whisper model for STT\n"
        "  --tokenizer <path>       MMS_FA tokenizer JSON\n"
        "  --whisper-cli <path>     Path to whisper-cli executable\n"
        "  --language <code>        Whisper language hint (default: auto)\n"
        "  --lead <ms>              Shift timestamps earlier by ms (default: 0)\n"
        "  --min-confidence <float> Drop lines with alignment confidence below\n"
        "                           this value (0-1, default: 0)\n"
        "  --recover-missing        Re-emit sung sections the lyrics omit\n"
        "  --method <a|b|c>         Alignment method (default: a, or b without lyrics)\n"
        "  --boost <float>          CTC non-blank boost (default: 5.0)\n"
        "  --gpu                    Use GPU acceleration (auto-detect CUDA/CoreML)\n"
        "  --provider <name>        Force provider: auto, cpu, cuda, coreml\n"
        "  --transcript <path>      Load a cached whisper JSON transcript\n"
        "  --verbose                Print detailed alignment info\n"
        "  -h, --help               Show this help\n\n"
        "Methods:\n"
        "  a    CTC forced alignment only (MMS multilingual, needs lyrics)\n"
        "  b    Audio-only: Whisper STT -> LRC (no lyrics needed)\n"
        "  c    Whisper STT + hint reconciliation (re-derive structure, fix typos)\n\n"
        "Providers (GPU acceleration):\n"
        "  auto   Detect best available (CUDA > CoreML > CPU)\n"
        "  cpu    CPU only\n"
        "  cuda   NVIDIA GPU (Linux/Windows, requires CUDA build)\n"
        "  coreml Apple GPU/ANE (macOS, requires CoreML build)\n"
    );
}

int main(int argc, char** argv) {
    std::string audio_path;
    std::string lyrics_path;
    std::string output_path;
    std::string ffmpeg_path;
    std::string whisper_cli_path;
    std::string model_a_path;
    std::string model_c_path;
    std::string tokenizer_path;
    std::string language = "auto";
    std::string method = "a";
    bool method_explicit = false;
    std::string transcript_path;
    std::string provider_str = "auto";
    float boost = 5.0f;
    int lead_ms = 0;
    float min_conf = 0.0f;
    bool verbose = false;
    bool recover_missing = false;
    std::vector<std::string> positionals;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-o" || arg == "--output") {
            if (!take_value(argc, argv, i, arg.c_str(), output_path)) return 1;
        } else if (arg == "--ffmpeg") {
            if (!take_value(argc, argv, i, arg.c_str(), ffmpeg_path)) return 1;
        } else if (arg == "--whisper-cli") {
            if (!take_value(argc, argv, i, arg.c_str(), whisper_cli_path)) return 1;
        } else if (arg == "--model-a") {
            if (!take_value(argc, argv, i, arg.c_str(), model_a_path)) return 1;
        } else if (arg == "--model-c") {
            if (!take_value(argc, argv, i, arg.c_str(), model_c_path)) return 1;
        } else if (arg == "--tokenizer") {
            if (!take_value(argc, argv, i, arg.c_str(), tokenizer_path)) return 1;
        } else if (arg == "--language") {
            if (!take_value(argc, argv, i, arg.c_str(), language)) return 1;
        } else if (arg == "--method") {
            if (!take_value(argc, argv, i, arg.c_str(), method)) return 1;
            method_explicit = true;
        } else if (arg == "--boost") {
            std::string value;
            if (!take_numeric_value(argc, argv, i, arg.c_str(), value) ||
                !parse_float_value(value, boost) || boost < 0.0f) {
                fprintf(stderr, "Error: --boost requires a finite non-negative number\n");
                return 1;
            }
        } else if (arg == "--gpu") {
            provider_str = "auto";
        } else if (arg == "--provider") {
            if (!take_value(argc, argv, i, arg.c_str(), provider_str)) return 1;
        } else if (arg == "--transcript") {
            if (!take_value(argc, argv, i, arg.c_str(), transcript_path)) return 1;
        } else if (arg == "--lead") {
            std::string value;
            if (!take_numeric_value(argc, argv, i, arg.c_str(), value) ||
                !parse_int_value(value, lead_ms)) {
                fprintf(stderr, "Error: --lead requires an integer\n");
                return 1;
            }
        } else if (arg == "--min-confidence") {
            std::string value;
            if (!take_numeric_value(argc, argv, i, arg.c_str(), value) ||
                !parse_float_value(value, min_conf) ||
                min_conf < 0.0f || min_conf > 1.0f) {
                fprintf(stderr, "Error: --min-confidence requires a number from 0 to 1\n");
                return 1;
            }
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--recover-missing") {
            recover_missing = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "-") {
            if (positionals.size() >= 2) {
                fprintf(stderr, "Error: too many positional arguments\n");
                return 1;
            }
            positionals.push_back(arg);
        } else if (!arg.empty() && arg[0] == '-') {
            fprintf(stderr, "Error: unknown option '%s'\n", arg.c_str());
            return 1;
        } else {
            if (positionals.size() >= 2) {
                fprintf(stderr, "Error: too many positional arguments\n");
                return 1;
            }
            positionals.push_back(arg);
        }
    }

    if (positionals.empty()) {
        fprintf(stderr, "Error: an audio file is required\n");
        print_usage();
        return 1;
    }
    if (positionals.size() > 1) lyrics_path = positionals[1];
    audio_path = positionals[0];

    if (method != "a" && method != "b" && method != "c") {
        fprintf(stderr, "Error: invalid method '%s' (expected a, b, or c)\n", method.c_str());
        return 1;
    }
    Provider provider = Provider::Auto;
    if (provider_str == "cpu") provider = Provider::CPU;
    else if (provider_str == "cuda") provider = Provider::CUDA;
    else if (provider_str == "coreml") provider = Provider::CoreML;
    else if (provider_str == "auto") provider = Provider::Auto;
    else {
        fprintf(stderr, "Error: invalid provider '%s' (expected auto, cpu, cuda, or coreml)\n",
                provider_str.c_str());
        return 1;
    }

    if (lyrics_path.empty()) {
        if (!method_explicit) {
            method = "b";
            fprintf(stderr, "No lyrics given - using method b (audio-only transcription)\n");
        } else if (method == "a" || method == "c") {
            fprintf(stderr, "Error: method '%s' requires a lyrics file\n", method.c_str());
            return 1;
        }
    } else if (method == "b") {
        fprintf(stderr, "Error: method b does not accept a lyrics file\n");
        return 1;
    }

    fs::path executable_dir(process::executable_directory(argc > 0 ? argv[0] : nullptr));
    if (model_a_path.empty())
        model_a_path = runtime_file(executable_dir, "models/mms_multilingual.onnx");
    if (model_c_path.empty())
        model_c_path = runtime_file(executable_dir, "models/ggml-large-v3.bin");
    if (tokenizer_path.empty())
        tokenizer_path = runtime_file(executable_dir, "models/mms_multilingual_tokenizer.json");
    if (ffmpeg_path.empty())
        ffmpeg_path = first_existing(executable_dir,
                                     {"ffmpeg", "ffmpeg.exe", "bin/ffmpeg", "bin/ffmpeg.exe"},
                                     "ffmpeg");
    if (whisper_cli_path.empty())
        whisper_cli_path = first_existing(
            executable_dir,
            {"bin/whisper-cli", "bin/whisper-cli.exe", "whisper-cli", "whisper-cli.exe"},
            "whisper-cli");

    if (output_path.empty()) {
        fs::path p(audio_path);
        output_path = p.stem().string() + ".lrc";
    }

    fprintf(stderr, "=== linetime ===\n");
    fprintf(stderr, "Audio:   %s\n", audio_path.c_str());
    fprintf(stderr, "Lyrics:  %s\n", lyrics_path.empty() ? "(none - audio-only)" : lyrics_path.c_str());
    fprintf(stderr, "Output:  %s\n", output_path.c_str());
    fprintf(stderr, "Method:  %s\n", method.c_str());
    fprintf(stderr, "Provider: %s\n\n", provider_str.c_str());

    fprintf(stderr, "[1/5] Loading audio...\n");
    utils::report_progress(5, "Loading audio");
    AudioBuffer audio = load_audio(audio_path, ffmpeg_path);
    if (audio.n_samples == 0) {
        fprintf(stderr, "Error: failed to load audio\n");
        return 1;
    }

    LyricsDocument lyrics;
    if (!lyrics_path.empty()) {
        fprintf(stderr, "[2/5] Parsing lyrics...\n");
        utils::report_progress(12, "Parsing lyrics");
        lyrics = parse_lyrics(lyrics_path);
        if (lyrics.total_lines == 0) {
            fprintf(stderr, "Error: no lyrics lines found\n");
            return 1;
        }
    } else {
        fprintf(stderr, "[2/5] Skipping lyrics (audio-only method b)\n");
    }

    std::vector<AlignedLine> ctc_result;
    std::vector<AlignedLine> transcribe_result;

    if (method == "a") {
        fprintf(stderr, "[3/5] Running CTC forced alignment (MMS_FA)...\n");
        utils::report_progress(20, "Loading alignment model");
        CTCAligner ctc;
        if (ctc.init(model_a_path, tokenizer_path, provider)) {
            CTCAlignerResult result = ctc.align(audio, lyrics, boost);
            if (result.success) {
                ctc_result = result.lines;
                fprintf(stderr, "  CTC: %zu lines aligned\n", ctc_result.size());
            } else {
                fprintf(stderr, "  CTC failed: %s\n", result.error.c_str());
            }
        } else {
            if (ctc.provider_unavailable()) return 1;
            fprintf(stderr, "  Failed to init CTC aligner (model not found?)\n");
        }
    } else {
        fprintf(stderr, "[3/5] Skipping CTC (method=%s)\n", method.c_str());
    }

    if (method == "c") {
        fprintf(stderr, "[5/5] Running Whisper STT + hint reconciliation...\n");
        utils::report_progress(35, "Transcribing audio");
        TranscriptionResult trans;
        if (!transcript_path.empty()) {
            fprintf(stderr, "  Loading cached transcript from %s\n", transcript_path.c_str());
            trans = load_transcription_json(transcript_path);
        } else {
            std::vector<std::string> environment =
                runtime_environment(executable_dir, whisper_cli_path);
            trans = transcribe_audio(audio, model_c_path, whisper_cli_path, language,
                                     provider, 16, environment,
                                     fs::path(audio_path).stem().string());
        }
        if (trans.success) {
            fprintf(stderr, "  Transcribed: %zu segments, language=%s\n", trans.segments.size(), trans.language.c_str());
            ReconcileResult rec = reconcile_lyrics(lyrics, trans, 0.5f, recover_missing);
            if (rec.success) {
                for (const auto& rl : rec.lines) {
                    AlignedLine al;
                    al.line_index = 0;
                    al.start_ms = std::max(0LL, (long long)rl.start_ms - lead_ms);
                    al.end_ms = std::max(0LL, (long long)rl.end_ms - lead_ms);
                    al.confidence = rl.confidence;
                    al.text = rl.text;
                    al.align_text = rl.align_text;
                    transcribe_result.push_back(al);
                }
                fprintf(stderr, "  Reconciled: %zu lines\n", transcribe_result.size());
                utils::report_progress(75, "Reconciling lyrics");

                int nref = refine_with_ctc(transcribe_result, audio, boost,
                                           model_a_path, tokenizer_path, provider, lead_ms);
                if (nref > 0)
                    fprintf(stderr, "  CTC refinement: %d/%zu lines re-timed\n", nref, transcribe_result.size());
                utils::report_progress(85, "Refining timings");
            } else {
                fprintf(stderr, "  Reconciliation failed: %s\n", rec.error.c_str());
            }
        } else {
            fprintf(stderr, "  Transcription failed: %s\n", trans.error.c_str());
        }
    }

    if (method == "b") {
        fprintf(stderr, "[3/5] Running Whisper STT (audio-only method b)...\n");
        utils::report_progress(20, "Transcribing audio");
        TranscriptionResult trans;
        if (!transcript_path.empty()) {
            fprintf(stderr, "  Loading cached transcript from %s\n", transcript_path.c_str());
            trans = load_transcription_json(transcript_path);
        } else {
            std::vector<std::string> environment =
                runtime_environment(executable_dir, whisper_cli_path);
            trans = transcribe_audio(audio, model_c_path, whisper_cli_path, language,
                                     provider, 16, environment,
                                     fs::path(audio_path).stem().string());
        }
        if (trans.success) {
            fprintf(stderr, "  Transcribed: %zu segments, language=%s\n", trans.segments.size(), trans.language.c_str());
            transcribe_result = segments_to_lines(trans);
            fprintf(stderr, "  Segments -> %zu LRC lines\n", transcribe_result.size());
            utils::report_progress(75, "Timing transcribed lines");
            if (!transcribe_result.empty()) {
                int nref = refine_with_ctc(transcribe_result, audio, boost,
                                           model_a_path, tokenizer_path, provider, lead_ms);
                fprintf(stderr, "  CTC refinement: %d/%zu lines re-timed\n", nref, transcribe_result.size());
                utils::report_progress(85, "Refining timings");
            }
        } else {
            fprintf(stderr, "  Transcription failed: %s\n", trans.error.c_str());
        }
    }

    fprintf(stderr, "[6/6] Writing output...\n");
    utils::report_progress(90, "Writing output");
    std::vector<AlignedLine> final_result;
    if (method == "a") final_result = ctc_result;
    else if (method == "c" || method == "b") final_result = transcribe_result;

    if (final_result.empty()) {
        fprintf(stderr, "Error: no alignment results to write\n");
        return 1;
    }

    if (min_conf > 0.0f) {
        int dropped = 0;
        std::vector<AlignedLine> kept;
        for (const auto& line : final_result) {
            if (!line.text.empty() && line.confidence < min_conf) {
                dropped++;
                continue;
            }
            kept.push_back(line);
        }
        if (kept.empty()) {
            fprintf(stderr, "Error: --min-confidence %.2f dropped all lines\n", min_conf);
            return 1;
        }
        if (dropped > 0)
            fprintf(stderr, "  Dropped %d low-confidence line(s) (conf < %.2f)\n",
                    dropped, min_conf);
        final_result = kept;
    }

    bool ok = write_lrc(output_path, final_result);
    if (!ok) {
        fprintf(stderr, "Error: failed to write LRC file\n");
        return 1;
    }

    utils::report_progress(100, "Done");
    fprintf(stderr, "\n=== Done ===\n");
    fprintf(stderr, "Output: %s\n", output_path.c_str());
    if (verbose) {
        fprintf(stderr, "\nAlignment details:\n");
        for (const auto& line : final_result) {
            fprintf(stderr, "  [%6.1f - %6.1f] conf=%.2f %s\n",
                    line.start_ms / 1000.0, line.end_ms / 1000.0,
                    line.confidence, line.text.c_str());
        }
    }

    return 0;
}

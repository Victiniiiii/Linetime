#include "audio.h"
#include "lyrics.h"
#include "ctc_aligner.h"
#include "whisper_aligner.h"
#include "merger.h"
#include "lrc_writer.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

void print_usage() {
    fprintf(stderr,
        "linetime - Lyric-Audio Timestamp Aligner\n\n"
        "Usage: linetime <audio_file> <lyrics_file> [options]\n\n"
        "Options:\n"
        "  -o, --output <path>      Output LRC file (default: <audio>.lrc)\n"
        "  --model-a <path>         MMS_FA ONNX model (default: models/mms_fa.onnx)\n"
        "  --model-b <path>         Whisper GGML model (default: models/ggml-base.bin)\n"
        "  --tokenizer <path>       Tokenizer JSON (default: models/tokenizer.json)\n"
        "  --language <code>        Whisper language hint (default: auto)\n"
        "  --method <a|b|both>      Alignment method (default: both)\n"
        "  --verbose                Print detailed alignment info\n"
        "  -h, --help               Show this help\n\n"
        "Methods:\n"
        "  a    CTC forced alignment only (MMS_FA)\n"
        "  b    Whisper DTW alignment only\n"
        "  both Hybrid: run both, merge best results (default)\n"
    );
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    std::string audio_path;
    std::string lyrics_path;
    std::string output_path;
    std::string model_a_path = "models/mms_fa.onnx";
    std::string model_b_path = "models/ggml-base.bin";
    std::string tokenizer_path = "models/tokenizer.json";
    std::string language = "auto";
    std::string method = "both";
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) output_path = argv[++i];
        } else if (arg == "--model-a") {
            if (i + 1 < argc) model_a_path = argv[++i];
        } else if (arg == "--model-b") {
            if (i + 1 < argc) model_b_path = argv[++i];
        } else if (arg == "--tokenizer") {
            if (i + 1 < argc) tokenizer_path = argv[++i];
        } else if (arg == "--language") {
            if (i + 1 < argc) language = argv[++i];
        } else if (arg == "--method") {
            if (i + 1 < argc) method = argv[++i];
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        } else if (audio_path.empty()) {
            audio_path = arg;
        } else if (lyrics_path.empty()) {
            lyrics_path = arg;
        }
    }

    if (audio_path.empty() || lyrics_path.empty()) {
        fprintf(stderr, "Error: both audio and lyrics files are required\n");
        print_usage();
        return 1;
    }

    if (output_path.empty()) {
        fs::path p(audio_path);
        output_path = p.stem().string() + ".lrc";
    }

    fprintf(stderr, "=== linetime ===\n");
    fprintf(stderr, "Audio:   %s\n", audio_path.c_str());
    fprintf(stderr, "Lyrics:  %s\n", lyrics_path.c_str());
    fprintf(stderr, "Output:  %s\n", output_path.c_str());
    fprintf(stderr, "Method:  %s\n\n", method.c_str());

    // Step 1: Load audio
    fprintf(stderr, "[1/5] Loading audio...\n");
    AudioBuffer audio = load_audio(audio_path);
    if (audio.n_samples == 0) {
        fprintf(stderr, "Error: failed to load audio\n");
        return 1;
    }

    // Step 2: Parse lyrics
    fprintf(stderr, "[2/5] Parsing lyrics...\n");
    LyricsDocument lyrics = parse_lyrics(lyrics_path);
    if (lyrics.total_lines == 0) {
        fprintf(stderr, "Error: no lyrics lines found\n");
        return 1;
    }

    // Step 3: Run alignment methods
    std::vector<AlignedLine> ctc_result;
    std::vector<AlignedLine> whisper_result;

    if (method == "a" || method == "both") {
        fprintf(stderr, "[3/5] Running CTC forced alignment (MMS_FA)...\n");
        CTCAligner ctc;
        if (ctc.init(model_a_path, tokenizer_path)) {
            CTCAlignerResult result = ctc.align(audio, lyrics);
            if (result.success) {
                ctc_result = result.lines;
                fprintf(stderr, "  CTC: %zu lines aligned\n", ctc_result.size());
            } else {
                fprintf(stderr, "  CTC failed: %s\n", result.error.c_str());
            }
        } else {
            fprintf(stderr, "  Failed to init CTC aligner (model not found?)\n");
        }
    } else {
        fprintf(stderr, "[3/5] Skipping CTC (method=%s)\n", method.c_str());
    }

    if (method == "b" || method == "both") {
        fprintf(stderr, "[4/5] Running Whisper DTW alignment...\n");
        WhisperAligner whisper;
        if (whisper.init(model_b_path)) {
            CTCAlignerResult result = whisper.align(audio, lyrics, language);
            if (result.success) {
                whisper_result = result.lines;
                fprintf(stderr, "  Whisper: %zu lines aligned\n", whisper_result.size());
            } else {
                fprintf(stderr, "  Whisper failed: %s\n", result.error.c_str());
            }
        } else {
            fprintf(stderr, "  Failed to init Whisper aligner (model not found?)\n");
        }
    } else {
        fprintf(stderr, "[4/5] Skipping Whisper (method=%s)\n", method.c_str());
    }

    // Step 4: Merge
    fprintf(stderr, "[5/5] Merging and writing output...\n");
    std::vector<AlignedLine> final_result;

    if (method == "both") {
        final_result = merge_alignment(ctc_result, whisper_result);
    } else if (method == "a") {
        final_result = ctc_result;
    } else {
        final_result = whisper_result;
    }

    if (final_result.empty()) {
        fprintf(stderr, "Error: no alignment results to write\n");
        return 1;
    }

    // Step 5: Write LRC
    bool ok = write_lrc(output_path, final_result);
    if (!ok) {
        fprintf(stderr, "Error: failed to write LRC file\n");
        return 1;
    }

    // Print summary
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

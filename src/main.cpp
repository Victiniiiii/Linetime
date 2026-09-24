#include "audio.h"
#include "lyrics.h"
#include "ctc_aligner.h"
#include "whisper_aligner.h"
#include "merger.h"
#include "lrc_writer.h"
#include "transcriber.h"
#include "reconcile.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

void print_usage() {
    fprintf(stderr,
        "linetime v1.2 - Lyric-Audio Timestamp Aligner\n\n"
        "Usage: linetime <audio_file> <lyrics_file> [options]\n"
        "       cat lyrics.txt | linetime <audio_file> - [options]\n\n"
        "Options:\n"
        "  -o, --output <path>      Output LRC file (default: <audio>.lrc)\n"
        "  --ffmpeg <path>          Path to ffmpeg binary (default: search PATH)\n"
        "  --model-a <path>         MMS_FA ONNX model (default: models/mms_fa.onnx)\n"
        "  --model-b <path>         Whisper GGML model (default: models/ggml-base.bin)\n"
        "  --model-c <path>         Whisper large model for STT (default: models/ggml-large-v3.bin)\n"
        "  --tokenizer <path>       Tokenizer JSON (default: models/tokenizer.json)\n"
        "  --language <code>        Whisper language hint (default: auto)\n"
        "  --method <a|b|c|both>    Alignment method (default: a)\n"
        "  --boost <float>          CTC non-blank boost (default: 5.0)\n"
        "  --gpu                    Use GPU acceleration (auto-detect CUDA/CoreML)\n"
        "  --provider <name>        Force provider: auto, cpu, cuda, coreml\n"
        "  --verbose                Print detailed alignment info\n"
        "  -h, --help               Show this help\n\n"
        "Methods:\n"
        "  a    CTC forced alignment only (MMS multilingual)\n"
        "  b    Whisper DTW alignment only\n"
        "  c    Whisper STT + hint reconciliation (re-derive structure, fix typos)\n"
        "  both Hybrid: run both CTC and Whisper DTW, merge best results\n\n"
        "Providers (GPU acceleration):\n"
        "  auto   Detect best available (CUDA > CoreML > CPU)\n"
        "  cpu    CPU only\n"
        "  cuda   NVIDIA GPU (Linux/Windows, requires CUDA build)\n"
        "  coreml Apple GPU/ANE (macOS, requires CoreML build)\n"
    );
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { print_usage(); return 0; }
    }

    if (argc < 3) {
        print_usage();
        return 1;
    }

    std::string audio_path;
    std::string lyrics_path;
    std::string output_path;
    std::string ffmpeg_path;
    std::string model_a_path = "models/mms_fa.onnx";
    std::string model_b_path = "models/ggml-base.bin";
    std::string model_c_path = "models/ggml-large-v3.bin";
    std::string tokenizer_path = "models/tokenizer.json";
    std::string language = "auto";
    std::string method = "a";
    std::string provider_str = "auto";
    float boost = 5.0f;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) output_path = argv[++i];
        } else if (arg == "--ffmpeg") {
            if (i + 1 < argc) ffmpeg_path = argv[++i];
        } else if (arg == "--model-a") {
            if (i + 1 < argc) model_a_path = argv[++i];
        } else if (arg == "--model-b") {
            if (i + 1 < argc) model_b_path = argv[++i];
        } else if (arg == "--model-c") {
            if (i + 1 < argc) model_c_path = argv[++i];
        } else if (arg == "--tokenizer") {
            if (i + 1 < argc) tokenizer_path = argv[++i];
        } else if (arg == "--language") {
            if (i + 1 < argc) language = argv[++i];
        } else if (arg == "--method") {
            if (i + 1 < argc) method = argv[++i];
        } else if (arg == "--boost") {
            if (i + 1 < argc) boost = std::stof(argv[++i]);
        } else if (arg == "--gpu") {
            provider_str = "auto";
        } else if (arg == "--provider") {
            if (i + 1 < argc) provider_str = argv[++i];
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

    // Parse provider
    Provider provider = Provider::Auto;
    if (provider_str == "cpu") provider = Provider::CPU;
    else if (provider_str == "cuda") provider = Provider::CUDA;
    else if (provider_str == "coreml") provider = Provider::CoreML;
    else provider = Provider::Auto;

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
    fprintf(stderr, "Method:  %s\n", method.c_str());
    fprintf(stderr, "Provider: %s\n\n", provider_str.c_str());

    // Step 1: Load audio
    fprintf(stderr, "[1/5] Loading audio...\n");
    AudioBuffer audio = load_audio(audio_path, ffmpeg_path);
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
    std::vector<AlignedLine> transcribe_result;

    if (method == "a" || method == "both") {
        fprintf(stderr, "[3/5] Running CTC forced alignment (MMS_FA)...\n");
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
            fprintf(stderr, "  Failed to init CTC aligner (model not found?)\n");
        }
    } else if (method == "c") {
        fprintf(stderr, "[3/5] Skipping CTC (method=c)\n");
    } else {
        fprintf(stderr, "[3/5] Skipping CTC (method=%s)\n", method.c_str());
    }

    if (method == "b" || method == "both") {
        fprintf(stderr, "[4/5] Running Whisper DTW alignment...\n");
        WhisperAligner whisper;
        if (whisper.init(model_b_path, provider)) {
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
    } else if (method == "c") {
        fprintf(stderr, "[4/5] Skipping Whisper DTW (method=c)\n");
    } else {
        fprintf(stderr, "[4/5] Skipping Whisper (method=%s)\n", method.c_str());
    }

    if (method == "c") {
        fprintf(stderr, "[5/5] Running Whisper STT + hint reconciliation...\n");
        TranscriptionResult trans = transcribe_audio(audio_path, model_c_path, language);
        if (trans.success) {
            fprintf(stderr, "  Transcribed: %zu segments, language=%s\n", trans.segments.size(), trans.language.c_str());
            ReconcileResult rec = reconcile_lyrics(lyrics, trans);
            if (rec.success) {
                // Convert ReconciledLine to AlignedLine
                for (const auto& rl : rec.lines) {
                    AlignedLine al;
                    al.line_index = 0; // will be sorted by time
                    al.start_ms = rl.start_ms;
                    al.end_ms = rl.end_ms;
                    al.confidence = rl.confidence;
                    al.text = rl.text;
                    transcribe_result.push_back(al);
                }
                fprintf(stderr, "  Reconciled: %zu lines\n", transcribe_result.size());
            } else {
                fprintf(stderr, "  Reconciliation failed: %s\n", rec.error.c_str());
            }
        } else {
            fprintf(stderr, "  Transcription failed: %s\n", trans.error.c_str());
        }
    }

    // Step 4: Merge / select result
    fprintf(stderr, "[6/6] Merging and writing output...\n");
    std::vector<AlignedLine> final_result;

    if (method == "both") {
        final_result = merge_alignment(ctc_result, whisper_result);
    } else if (method == "a") {
        final_result = ctc_result;
    } else if (method == "b") {
        final_result = whisper_result;
    } else if (method == "c") {
        final_result = transcribe_result;
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

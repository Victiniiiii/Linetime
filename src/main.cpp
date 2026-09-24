#include "audio.h"
#include "lyrics.h"
#include "ctc_aligner.h"
#include "lrc_writer.h"
#include "transcriber.h"
#include "reconcile.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// Convert whisper segments into LRC lines (method b). One line per non-empty
// segment, whitespace/paragraph markers stripped.
static std::vector<AlignedLine> segments_to_lines(const TranscriptionResult& trans) {
    std::vector<AlignedLine> out;
    for (const auto& seg : trans.segments) {
        std::string text = seg.text;
        size_t b = text.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        size_t e = text.find_last_not_of(" \t\r\n");
        text = text.substr(b, e - b + 1);
        // Collapse padding/newlines left by whisper paragraph markers
        for (size_t i = 0; i < text.size(); i++) {
            if (text[i] == '\n' || text[i] == '\r' || text[i] == '\t') text[i] = ' ';
        }
        bool in_space = false;
        std::string clean;
        clean.reserve(text.size());
        for (char c : text) {
            if (c == ' ') { if (!in_space) { clean += ' '; in_space = true; } }
            else { clean += c; in_space = false; }
        }
        if (clean.empty()) continue;

        float conf_sum = 0.0f;
        int conf_n = 0;
        for (const auto& w : seg.words) { conf_sum += w.prob; conf_n++; }

        AlignedLine al;
        al.line_index = (int)out.size();
        al.start_ms = seg.start_ms;
        al.end_ms = seg.end_ms;
        al.confidence = conf_n > 0 ? conf_sum / conf_n : 0.0f;
        al.text = clean;
        al.align_text = clean;
        out.push_back(std::move(al));
    }
    return out;
}

// Re-time whisper-timed lines with MMS CTC forced alignment where confident.
// CTC times are a far more reliable oracle than whisper word starts (whisper
// marks first words of segments early). Keeps whisper times when CTC is
// unavailable or not confident (>0.10). Returns number of re-timed lines.
static int refine_with_ctc(std::vector<AlignedLine>& lines,
                           const AudioBuffer& audio,
                           float boost,
                           const std::string& model_a_path,
                           const std::string& tokenizer_path,
                           Provider provider,
                           int lead_ms) {
    LyricsDocument refined;
    std::vector<size_t> keep_idx; // lines[] index for each refined lyric line
    for (size_t i = 0; i < lines.size(); i++) {
        if (lines[i].text.empty()) continue; // blank paragraph marker
        LyricLine ll;
        ll.index = (int)refined.lines.size();
        ll.text = lines[i].align_text.empty() ? lines[i].text : lines[i].align_text;
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

void print_usage() {
    fprintf(stderr,
        "linetime v1.2 - Lyric-Audio Timestamp Aligner\n\n"
        "Usage: linetime <audio_file> [lyrics_file] [options]\n"
        "       cat lyrics.txt | linetime <audio_file> - [options]\n"
        "       linetime <audio_file> [options]     (no lyrics -> method b)\n\n"
        "Options:\n"
        "  -o, --output <path>      Output LRC file (default: <audio>.lrc)\n"
        "  --ffmpeg <path>          Path to ffmpeg binary (default: search PATH)\n"
        "  --model-a <path>         MMS_FA ONNX model (default: models/mms_multilingual.onnx)\n"
        "  --model-c <path>         Whisper large model for STT (default: models/ggml-large-v3.bin)\n"
        "  --tokenizer <path>       Tokenizer JSON (default: models/mms_multilingual_tokenizer.json)\n"
        "  --language <code>        Whisper language hint (default: auto)\n"
        "  --lead <ms>              Shift timestamps earlier by ms (default: 0)\n"
        "  --min-confidence <float> Drop lines with alignment confidence below\n"
        "                           this value (0-1, default: 0)\n"
        "  --recover-missing       Re-emit sung sections the lyrics omit (e.g.\n"
        "                           a non-repeated chorus). Off by default: whisper\n"
        "                           hallucination chant-tails can be re-added.\n"
        "  --method <a|b|c>          Alignment method (default: a, or b without lyrics)\n"
        "  --boost <float>          CTC non-blank boost (default: 5.0)\n"
        "  --gpu                    Use GPU acceleration (auto-detect CUDA/CoreML)\n"
        "  --provider <name>        Force provider: auto, cpu, cuda, coreml\n"
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
    std::string model_a_path = "models/mms_multilingual.onnx";
    std::string model_c_path = "models/ggml-large-v3.bin";
    std::string tokenizer_path = "models/mms_multilingual_tokenizer.json";
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

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) output_path = argv[++i];
        } else if (arg == "--ffmpeg") {
            if (i + 1 < argc) ffmpeg_path = argv[++i];
        } else if (arg == "--model-a") {
            if (i + 1 < argc) model_a_path = argv[++i];
        } else if (arg == "--model-c") {
            if (i + 1 < argc) model_c_path = argv[++i];
        } else if (arg == "--tokenizer") {
            if (i + 1 < argc) tokenizer_path = argv[++i];
        } else if (arg == "--language") {
            if (i + 1 < argc) language = argv[++i];
        } else if (arg == "--method") {
            if (i + 1 < argc) { method = argv[++i]; method_explicit = true; }
        } else if (arg == "--boost") {
            if (i + 1 < argc) boost = std::stof(argv[++i]);
        } else if (arg == "--gpu") {
            provider_str = "auto";
        } else if (arg == "--provider") {
            if (i + 1 < argc) provider_str = argv[++i];
        } else if (arg == "--transcript") {
            if (i + 1 < argc) transcript_path = argv[++i];
        } else if (arg == "--lead") {
            if (i + 1 < argc) lead_ms = atoi(argv[++i]);
        } else if (arg == "--min-confidence") {
            if (i + 1 < argc) min_conf = std::stof(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--recover-missing") {
            recover_missing = true;
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

    if (audio_path.empty()) {
        fprintf(stderr, "Error: an audio file is required\n");
        print_usage();
        return 1;
    }

    // No lyrics -> audio-only transcription (method b). Explicit method a/c
    // without lyrics is an error; method b ignores a lyrics file if given.
    if (lyrics_path.empty()) {
        if (!method_explicit) {
            method = "b";
            fprintf(stderr, "No lyrics given - using method b (audio-only transcription)\n");
        } else if (method == "a" || method == "c") {
            fprintf(stderr, "Error: method '%s' requires a lyrics file\n", method.c_str());
            fprintf(stderr, "       Omit the lyrics file (or pass --method b) for audio-only LRC\n");
            return 1;
        }
    } else if (method == "b") {
        fprintf(stderr, "Method b ignores the lyrics file (%s); use method a/c for lyrics alignment\n",
                lyrics_path.c_str());
        lyrics_path.clear();
    }

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

    // Step 1: Load audio
    fprintf(stderr, "[1/5] Loading audio...\n");
    AudioBuffer audio = load_audio(audio_path, ffmpeg_path);
    if (audio.n_samples == 0) {
        fprintf(stderr, "Error: failed to load audio\n");
        return 1;
    }

    // Step 2: Parse lyrics (audio-only mode has none)
    LyricsDocument lyrics;
    if (!lyrics_path.empty()) {
        fprintf(stderr, "[2/5] Parsing lyrics...\n");
        lyrics = parse_lyrics(lyrics_path);
        if (lyrics.total_lines == 0) {
            fprintf(stderr, "Error: no lyrics lines found\n");
            return 1;
        }
    } else {
        fprintf(stderr, "[2/5] Skipping lyrics (audio-only method b)\n");
    }

    // Step 3: Run alignment methods
    std::vector<AlignedLine> ctc_result;
    std::vector<AlignedLine> transcribe_result;

    if (method == "a") {
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
    } else {
        fprintf(stderr, "[3/5] Skipping CTC (method=%s)\n", method.c_str());
    }

    if (method == "c") {
        fprintf(stderr, "[5/5] Running Whisper STT + hint reconciliation...\n");
        TranscriptionResult trans;
        if (!transcript_path.empty()) {
            fprintf(stderr, "  Loading cached transcript from %s\n", transcript_path.c_str());
            trans = load_transcription_json(transcript_path);
        } else {
            trans = transcribe_audio(audio_path, model_c_path, language);
        }
        if (trans.success) {
            fprintf(stderr, "  Transcribed: %zu segments, language=%s\n", trans.segments.size(), trans.language.c_str());
            ReconcileResult rec = reconcile_lyrics(lyrics, trans, 0.5f, recover_missing);
            if (rec.success) {
                // Convert ReconciledLine to AlignedLine (whisper timings)
                for (const auto& rl : rec.lines) {
                    AlignedLine al;
                    al.line_index = 0; // will be sorted by time
                    al.start_ms = std::max(0LL, (long long)rl.start_ms - lead_ms);
                    al.end_ms = std::max(0LL, (long long)rl.end_ms - lead_ms);
                    al.confidence = rl.confidence;
                    al.text = rl.text;
                    al.align_text = rl.align_text;
                    transcribe_result.push_back(al);
                }
                fprintf(stderr, "  Reconciled: %zu lines\n", transcribe_result.size());

                // CTC timing refinement: MMS forced alignment on the reconciled
                // text is a far more reliable timing oracle than whisper word
                // starts (whisper marks first words of segments early). Use CTC
                // times for every line it can align; keep whisper times for the
                // rest.
                int nref = refine_with_ctc(transcribe_result, audio, boost,
                                           model_a_path, tokenizer_path, provider, lead_ms);
                if (nref > 0)
                    fprintf(stderr, "  CTC refinement: %d/%zu lines re-timed\n", nref, transcribe_result.size());
            } else {
                fprintf(stderr, "  Reconciliation failed: %s\n", rec.error.c_str());
            }
        } else {
            fprintf(stderr, "  Transcription failed: %s\n", trans.error.c_str());
        }
    }

    if (method == "b") {
        fprintf(stderr, "[3/5] Running Whisper STT (audio-only method b)...\n");
        TranscriptionResult trans;
        if (!transcript_path.empty()) {
            fprintf(stderr, "  Loading cached transcript from %s\n", transcript_path.c_str());
            trans = load_transcription_json(transcript_path);
        } else {
            trans = transcribe_audio(audio_path, model_c_path, language);
        }
        if (trans.success) {
            fprintf(stderr, "  Transcribed: %zu segments, language=%s\n", trans.segments.size(), trans.language.c_str());
            transcribe_result = segments_to_lines(trans);
            fprintf(stderr, "  Segments -> %zu LRC lines\n", transcribe_result.size());
            if (!transcribe_result.empty()) {
                int nref = refine_with_ctc(transcribe_result, audio, boost,
                                           model_a_path, tokenizer_path, provider, lead_ms);
                fprintf(stderr, "  CTC refinement: %d/%zu lines re-timed\n", nref, transcribe_result.size());
            }
        } else {
            fprintf(stderr, "  Transcription failed: %s\n", trans.error.c_str());
        }
    }

    // Step 4: Select result
    fprintf(stderr, "[6/6] Writing output...\n");
    std::vector<AlignedLine> final_result;

    if (method == "a") {
        final_result = ctc_result;
    } else if (method == "c" || method == "b") {
        final_result = transcribe_result;
    } else {
        fprintf(stderr, "Error: unknown method '%s'\n", method.c_str());
        return 1;
    }

    if (final_result.empty()) {
        fprintf(stderr, "Error: no alignment results to write\n");
        return 1;
    }

    // Confidence-based filtering: drop low-confidence lines (blank paragraph
    // separators are always kept).
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

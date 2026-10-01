#include "audio.h"
#include "child_process.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static std::string find_ffmpeg() {
    std::filesystem::path base(process::executable_directory());
    std::vector<std::filesystem::path> candidates = {
        base / "ffmpeg",
        base / "ffmpeg.exe",
        base / "bin" / "ffmpeg",
        base / "bin" / "ffmpeg.exe"
    };
    for (const auto& candidate : candidates) {
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error) && !error)
            return candidate.string();
    }
    return "ffmpeg";
}

AudioBuffer load_audio(const std::string& path, const std::string& ffmpeg_path,
                       bool separate_vocals) {
    AudioBuffer result;
    std::string ffmpeg = ffmpeg_path.empty() ? find_ffmpeg() : ffmpeg_path;
    if (ffmpeg.empty()) {
        fprintf(stderr, "[audio] ffmpeg not found. Use --ffmpeg to specify the path.\n");
        return result;
    }

    std::vector<std::string> arguments = {
        ffmpeg,
        "-nostdin",
        "-i", path,
        "-f", "f32le",
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_f32le",
        "-hide_banner",
        "-loglevel", "error",
        "-"
    };

    if (separate_vocals) {
        // Sum the channels instead of taking one, which keeps whatever is
        // positioned centrally and cancels the wide stereo parts. On the test set
        // the centre sat 10-15 dB above the sides, so the accompaniment really does
        // drop back. It is inserted before the format conversion so it works on the
        // original channel layout.
        arguments.insert(arguments.end() - 1,
                         {"-af", "pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1"});
    }
    process::Result execution = process::run(arguments);
    if (!execution.started) {
        fprintf(stderr, "[audio] Failed to run ffmpeg for %s: %s\n",
                path.c_str(), execution.error.c_str());
        return result;
    }
    if (execution.exit_code != 0) {
        fprintf(stderr, "[audio] ffmpeg exited with code %d for %s\n",
                execution.exit_code, path.c_str());
        if (!execution.stderr_data.empty())
            fprintf(stderr, "%s\n", execution.stderr_data.c_str());
        return result;
    }
    if (!execution.error.empty()) {
        fprintf(stderr, "[audio] ffmpeg output capture failed: %s\n",
                execution.error.c_str());
        return result;
    }
    if (execution.stdout_data.size() % sizeof(float) != 0) {
        fprintf(stderr, "[audio] ffmpeg returned an incomplete float sample for %s\n",
                path.c_str());
        return result;
    }

    size_t sample_count = execution.stdout_data.size() / sizeof(float);
    if (sample_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "[audio] ffmpeg returned too many samples for %s\n", path.c_str());
        return result;
    }
    result.samples.resize(sample_count);
    if (sample_count > 0)
        std::memcpy(result.samples.data(), execution.stdout_data.data(),
                    sample_count * sizeof(float));
    result.sample_rate = 16000;
    result.n_samples = static_cast<int>(sample_count);
    result.duration_sec = static_cast<double>(result.n_samples) / 16000.0;

    fprintf(stderr, "[audio] Loaded %s: %d samples, %.1f sec, 16kHz mono%s\n",
            path.c_str(), result.n_samples, result.duration_sec,
            separate_vocals ? " (centre channel only)" : "");
    return result;
}

AudioBuffer load_wav_16k_mono(const std::string& path, const std::string& ffmpeg_path,
                              bool separate_vocals) {
    return load_audio(path, ffmpeg_path, separate_vocals);
}

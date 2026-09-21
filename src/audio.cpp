#include "audio.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>

static std::string find_ffmpeg() {
    const char* paths[] = {
        "ffmpeg",
        "./ffmpeg",
#ifdef _WIN32
        ".\\ffmpeg.exe",
        "C:\\ffmpeg\\bin\\ffmpeg.exe",
#else
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
#endif
    };
    for (const char* p : paths) {
#ifdef _WIN32
        std::string cmd = std::string(p) + " -version >nul 2>nul";
#else
        std::string cmd = std::string(p) + " -version 2>/dev/null";
#endif
        if (system(cmd.c_str()) == 0) return p;
    }
    return "";
}

AudioBuffer load_audio(const std::string& path, const std::string& ffmpeg_path) {
    AudioBuffer result;

    std::string ffmpeg = ffmpeg_path.empty() ? find_ffmpeg() : ffmpeg_path;
    if (ffmpeg.empty()) {
        fprintf(stderr, "[audio] ffmpeg not found. Use --ffmpeg to specify the path.\n");
        return result;
    }

    std::string cmd = ffmpeg + " -i \"" + path +
        "\" -f f32le -ar 16000 -ac 1 -acodec pcm_f32le -hide_banner -loglevel error - 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        fprintf(stderr, "[audio] Failed to run ffmpeg for %s\n", path.c_str());
        return result;
    }

    std::vector<float> all_samples;
    constexpr size_t BUF_SAMPLES = 4096;
    float buf[BUF_SAMPLES];

    while (true) {
        size_t n = fread(buf, sizeof(float), BUF_SAMPLES, pipe);
        if (n == 0) break;
        all_samples.insert(all_samples.end(), buf, buf + n);
    }

    int rc = pclose(pipe);
    if (rc != 0) {
        fprintf(stderr, "[audio] ffmpeg exited with code %d for %s\n", rc, path.c_str());
        all_samples.clear();
    }

    result.samples = std::move(all_samples);
    result.sample_rate = 16000;
    result.n_samples = result.samples.size();
    result.duration_sec = static_cast<double>(result.n_samples) / 16000.0;

    fprintf(stderr, "[audio] Loaded %s: %d samples, %.1f sec, 16kHz mono\n",
            path.c_str(), result.n_samples, result.duration_sec);
    return result;
}

AudioBuffer load_wav_16k_mono(const std::string& path, const std::string& ffmpeg_path) {
    return load_audio(path, ffmpeg_path);
}

#include "audio.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

#include <cstdio>
#include <cstring>
#include <algorithm>

static void log_error(const char* msg, int err) {
    char buf[256];
    av_strerror(err, buf, sizeof(buf));
    fprintf(stderr, "[audio] %s: %s\n", msg, buf);
}

AudioBuffer load_audio(const std::string& path) {
    AudioBuffer result;

    AVFormatContext* fmt_ctx = nullptr;
    int ret = avformat_open_input(&fmt_ctx, path.c_str(), nullptr, nullptr);
    if (ret < 0) {
        log_error("Failed to open audio file", ret);
        return result;
    }

    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        log_error("Failed to find stream info", ret);
        avformat_close_input(&fmt_ctx);
        return result;
    }

    // Find the best audio stream
    int audio_stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (audio_stream_idx < 0) {
        fprintf(stderr, "[audio] No audio stream found in %s\n", path.c_str());
        avformat_close_input(&fmt_ctx);
        return result;
    }

    AVStream* audio_stream = fmt_ctx->streams[audio_stream_idx];
    AVCodecParameters* codecpar = audio_stream->codecpar;

    // Find decoder
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        fprintf(stderr, "[audio] Unsupported codec\n");
        avformat_close_input(&fmt_ctx);
        return result;
    }

    AVCodecContext* dec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(dec_ctx, codecpar);
    dec_ctx->request_sample_fmt = AV_SAMPLE_FMT_FLT; // Request float output

    ret = avcodec_open2(dec_ctx, codec, nullptr);
    if (ret < 0) {
        log_error("Failed to open codec", ret);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx);
        return result;
    }

    // Set up resampler: convert to 16kHz mono float32
    SwrContext* swr = swr_alloc();
    av_opt_set_int(swr, "in_channel_count", dec_ctx->channels, 0);
    av_opt_set_int(swr, "in_sample_rate", dec_ctx->sample_rate, 0);
    av_opt_set_sample_fmt(swr, "in_sample_fmt", dec_ctx->sample_fmt, 0);

    av_opt_set_int(swr, "out_channel_count", 1, 0);
    av_opt_set_int(swr, "out_sample_rate", 16000, 0);
    av_opt_set_sample_fmt(swr, "out_sample_fmt", AV_SAMPLE_FMT_FLT, 0);

    ret = swr_init(swr);
    if (ret < 0) {
        log_error("Failed to init resampler", ret);
        swr_free(&swr);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx);
        return result;
    }

    // Decode and resample
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    std::vector<float> all_samples;
    int64_t total_frames = 0;

    while (av_read_frame(fmt_ctx, packet) >= 0) {
        if (packet->stream_index != audio_stream_idx) {
            av_packet_unref(packet);
            continue;
        }

        ret = avcodec_send_packet(dec_ctx, packet);
        av_packet_unref(packet);
        if (ret < 0) continue;

        while (avcodec_receive_frame(dec_ctx, frame) == 0) {
            // Calculate output frames
            int64_t out_frames = av_rescale_rnd(
                swr_get_delay(swr, dec_ctx->sample_rate) + frame->nb_samples,
                16000, dec_ctx->sample_rate, AV_ROUND_UP);

            size_t old_size = all_samples.size();
            all_samples.resize(old_size + out_frames);
            float* out_ptr = all_samples.data() + old_size;

            int converted = swr_convert(swr, (uint8_t**)&out_ptr, out_frames,
                                        (const uint8_t**)frame->data, frame->nb_samples);
            if (converted > 0) {
                all_samples.resize(old_size + converted);
                total_frames += converted;
            } else {
                all_samples.resize(old_size);
            }
        }
    }

    // Flush decoder
    avcodec_send_packet(dec_ctx, nullptr);
    while (avcodec_receive_frame(dec_ctx, frame) == 0) {
        int64_t out_frames = av_rescale_rnd(
            swr_get_delay(swr, dec_ctx->sample_rate) + frame->nb_samples,
            16000, dec_ctx->sample_rate, AV_ROUND_UP);

        size_t old_size = all_samples.size();
        all_samples.resize(old_size + out_frames);
        float* out_ptr = all_samples.data() + old_size;

        int converted = swr_convert(swr, (uint8_t**)&out_ptr, out_frames,
                                    (const uint8_t**)frame->data, frame->nb_samples);
        if (converted > 0) {
            all_samples.resize(old_size + converted);
            total_frames += converted;
        } else {
            all_samples.resize(old_size);
        }
    }

    // Normalize to [-1, 1] range if needed
    float max_val = 0.0f;
    for (float s : all_samples) max_val = std::max(max_val, std::abs(s));
    if (max_val > 1.0f) {
        for (float& s : all_samples) s /= max_val;
    }

    result.samples = std::move(all_samples);
    result.sample_rate = 16000;
    result.n_samples = result.samples.size();
    result.duration_sec = static_cast<double>(result.n_samples) / 16000.0;

    av_frame_free(&frame);
    av_packet_free(&packet);
    swr_free(&swr);
    avcodec_free_context(&dec_ctx);
    avformat_close_input(&fmt_ctx);

    fprintf(stderr, "[audio] Loaded %s: %d samples, %.1f sec, 16kHz mono\n",
            path.c_str(), result.n_samples, result.duration_sec);
    return result;
}

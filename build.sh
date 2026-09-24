#!/bin/bash
# Build linetime (Manual build from AGENTS.md)
set -e
cd "$(dirname "$0")"
g++ -O3 -DNDEBUG -std=c++17 -Wno-unused-result \
  -DORT_CUDA \
  -Isrc -Ivendor -Ivendor/whisper.cpp/include -Ivendor/whisper.cpp/ggml/include \
  -Ivendor/onnxruntime/include/onnxruntime/core/session \
  -o linetime \
  src/main.cpp src/audio.cpp src/lyrics.cpp \
  src/ctc_aligner.cpp src/whisper_aligner.cpp \
  src/merger.cpp src/lrc_writer.cpp \
  src/transcriber.cpp src/reconcile.cpp \
  build-static/src/libwhisper.a \
  build-static/ggml/src/libggml.a build-static/ggml/src/libggml-base.a build-static/ggml/src/libggml-cpu.a \
  -Lvendor/onnxruntime/lib -lonnxruntime -lonnxruntime_providers_cuda -lonnxruntime_providers_shared \
  -Wl,-rpath,'$ORIGIN/lib' -Wl,--allow-multiple-definition \
  -lpthread -ldl -lm -lgomp -lz
echo "built linetime"
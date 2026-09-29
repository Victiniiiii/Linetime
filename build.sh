#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${LINETIME_BUILD_DIR:-${ROOT_DIR}/build}"
ONNXRUNTIME_DIR="${LINETIME_ONNXRUNTIME_DIR:-${ROOT_DIR}/vendor/onnxruntime}"
JOBS="${LINETIME_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2')}"

if [ ! -f "${ONNXRUNTIME_DIR}/include/onnxruntime/core/session/onnxruntime_c_api.h" ]; then
    printf 'ONNX Runtime headers not found in %s\n' "${ONNXRUNTIME_DIR}" >&2
    exit 1
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLINETIME_ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}" \
    -DLINETIME_ONNXRUNTIME_STATIC=OFF \
    -DLINETIME_ENABLE_CUDA=ON \
    -DLINETIME_ENABLE_COREML=OFF \
    -DLINETIME_BUILD_WHISPER_CLI=ON
cmake --build "${BUILD_DIR}" --target linetime whisper-cli --parallel "${JOBS}"

if [ ! -x "${BUILD_DIR}/bin/linetime" ] || [ ! -x "${BUILD_DIR}/bin/whisper-cli" ]; then
    printf 'Build did not produce both linetime and whisper-cli\n' >&2
    exit 1
fi

install -m 0755 "${BUILD_DIR}/bin/linetime" "${ROOT_DIR}/linetime"
printf 'Built %s\n' "${ROOT_DIR}/linetime"
printf 'Built %s\n' "${BUILD_DIR}/bin/whisper-cli"

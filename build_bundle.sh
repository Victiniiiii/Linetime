#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${LINETIME_BUNDLE_BUILD_DIR:-${ROOT_DIR}/build-bundle}"
DIST_DIR="${LINETIME_DIST_DIR:-${ROOT_DIR}/dist}"
ONNXRUNTIME_DIR="${LINETIME_ONNXRUNTIME_DIR:-${ROOT_DIR}/vendor/onnxruntime}"
CUDA_DIR="${LINETIME_CUDA_DIR:-${ROOT_DIR}/vendor/cuda-12.6/lib}"
JOBS="${LINETIME_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2')}"

if [ ! -f "${ONNXRUNTIME_DIR}/include/onnxruntime/core/session/onnxruntime_c_api.h" ]; then
    printf 'ONNX Runtime headers not found in %s\n' "${ONNXRUNTIME_DIR}" >&2
    exit 1
fi

CUDA_REQUESTED="${LINETIME_ENABLE_CUDA:-auto}"
CUDA_ENABLED=0
if [ "${CUDA_REQUESTED}" = "1" ] || { [ "${CUDA_REQUESTED}" = "auto" ] && { [ -e "${ONNXRUNTIME_DIR}/lib/libonnxruntime_providers_cuda.so" ] || [ -e "${ONNXRUNTIME_DIR}/onnxruntime_providers_cuda.dll" ]; }; }; then
    CUDA_ENABLED=1
fi

COREML_REQUESTED="${LINETIME_ENABLE_COREML:-auto}"
COREML_ENABLED=0
if [ "${COREML_REQUESTED}" = "1" ] || { [ "${COREML_REQUESTED}" = "auto" ] && [ "$(uname -s)" = "Darwin" ]; }; then
    COREML_ENABLED=1
fi
if [ "${CUDA_ENABLED}" -eq 1 ] && [ "${COREML_ENABLED}" -eq 1 ]; then
    printf 'CUDA and CoreML are mutually exclusive, pick one\n' >&2
    exit 1
fi

CMAKE_EXTRA_ARGS=()
if [ "${CUDA_ENABLED}" -eq 1 ]; then
    CMAKE_EXTRA_ARGS+=("-DLINETIME_ONNXRUNTIME_STATIC=OFF" "-DLINETIME_ENABLE_CUDA=ON" "-DLINETIME_ENABLE_COREML=OFF")
elif [ "${COREML_ENABLED}" -eq 1 ]; then
    CMAKE_EXTRA_ARGS+=("-DLINETIME_ONNXRUNTIME_STATIC=ON" "-DLINETIME_ENABLE_CUDA=OFF" "-DLINETIME_ENABLE_COREML=ON")
else
    CMAKE_EXTRA_ARGS+=("-DLINETIME_ONNXRUNTIME_STATIC=ON" "-DLINETIME_ENABLE_CUDA=OFF" "-DLINETIME_ENABLE_COREML=OFF")
fi

printf '=== Linetime bundle build ===\n'
printf 'Build directory: %s\n' "${BUILD_DIR}"
printf 'ONNX Runtime: %s\n' "${ONNXRUNTIME_DIR}"
printf 'CUDA bundle: %s\n' "${CUDA_ENABLED}"
printf 'CoreML bundle: %s\n' "${COREML_ENABLED}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLINETIME_ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}" \
    -DLINETIME_BUILD_WHISPER_CLI=ON \
    "${CMAKE_EXTRA_ARGS[@]}"
cmake --build "${BUILD_DIR}" --target linetime whisper-cli --parallel "${JOBS}"

if [ ! -x "${BUILD_DIR}/bin/linetime" ]; then
    printf 'Linetime build did not produce %s\n' "${BUILD_DIR}/bin/linetime" >&2
    exit 1
fi
if [ ! -x "${BUILD_DIR}/bin/whisper-cli" ]; then
    printf 'whisper-cli build did not produce %s\n' "${BUILD_DIR}/bin/whisper-cli" >&2
    exit 1
fi

rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}/models" "${DIST_DIR}/bin"
install -m 0755 "${BUILD_DIR}/bin/linetime" "${DIST_DIR}/linetime"
install -m 0755 "${BUILD_DIR}/bin/whisper-cli" "${DIST_DIR}/whisper-cli"
install -m 0755 "${BUILD_DIR}/bin/whisper-cli" "${DIST_DIR}/bin/whisper-cli"

FFMPEG_SOURCE="${LINETIME_FFMPEG:-}"
if [ -z "${FFMPEG_SOURCE}" ] && [ -x "${ROOT_DIR}/ffmpeg" ]; then
    FFMPEG_SOURCE="${ROOT_DIR}/ffmpeg"
fi
if [ -z "${FFMPEG_SOURCE}" ] && command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_SOURCE="$(command -v ffmpeg)"
fi
if [ -z "${FFMPEG_SOURCE}" ] || [ ! -x "${FFMPEG_SOURCE}" ]; then
    printf 'ffmpeg is required to create a complete bundle\n' >&2
    exit 1
fi
install -m 0755 "${FFMPEG_SOURCE}" "${DIST_DIR}/ffmpeg"
if ! "${DIST_DIR}/ffmpeg" -version >/dev/null 2>&1; then
    printf 'Bundled ffmpeg could not be executed\n' >&2
    exit 1
fi

copy_glob() {
    local directory="$1"
    local pattern="$2"
    local source
    [ -d "${directory}" ] || return 0
    for source in "${directory}"/${pattern}; do
        [ -e "${source}" ] || continue
        cp -L "${source}" "${DIST_DIR}/lib/$(basename "${source}")"
    done
}

if [ "${CUDA_ENABLED}" -eq 1 ]; then
    mkdir -p "${DIST_DIR}/lib"
    copy_glob "${ONNXRUNTIME_DIR}/lib" 'libonnxruntime.so*'
    copy_glob "${ONNXRUNTIME_DIR}/lib" 'libonnxruntime_providers_cuda.so*'
    copy_glob "${ONNXRUNTIME_DIR}/lib" 'libonnxruntime_providers_shared.so*'
    copy_glob "${ONNXRUNTIME_DIR}" 'onnxruntime.dll'
    copy_glob "${ONNXRUNTIME_DIR}" 'onnxruntime_providers_cuda.dll'
    copy_glob "${ONNXRUNTIME_DIR}" 'onnxruntime_providers_shared.dll'
    for cuda_directory in "${CUDA_DIR}" /usr/local/cuda/lib64 /usr/local/cuda/lib64/lib /usr/local/cuda/targets/x86_64-linux/lib /usr/lib/x86_64-linux-gnu; do
        copy_glob "${cuda_directory}" 'libcudart.so*'
        copy_glob "${cuda_directory}" 'libcublas.so*'
        copy_glob "${cuda_directory}" 'libcublasLt.so*'
        copy_glob "${cuda_directory}" 'libcurand.so*'
        copy_glob "${cuda_directory}" 'libcufft.so*'
        copy_glob "${cuda_directory}" 'libcudnn*.so*'
        copy_glob "${cuda_directory}" 'cudart64*.dll'
        copy_glob "${cuda_directory}" 'cublas64*.dll'
        copy_glob "${cuda_directory}" 'cublasLt64*.dll'
        copy_glob "${cuda_directory}" 'curand64*.dll'
        copy_glob "${cuda_directory}" 'cufft64*.dll'
        copy_glob "${cuda_directory}" 'cudnn*.dll'
    done
    # GLIBCUDA needs the cuBLASLt/cuDNN versions that match the runtime the
    # provider was linked against, so require one match per component.
    for component in cudart cublas cublasLt curand cufft cudnn; do
        if ! compgen -G "${DIST_DIR}/lib/*${component}*" >/dev/null 2>&1; then
            printf 'Required GPU runtime component is missing: %s\n' "${component}" >&2
            exit 1
        fi
    done
    if ! compgen -G "${DIST_DIR}/lib/*onnxruntime_providers_cuda*" >/dev/null 2>&1; then
        printf 'Required GPU runtime library is missing: onnxruntime_providers_cuda\n' >&2
        exit 1
    fi
fi

for model in \
    models/mms_multilingual.onnx \
    models/mms_multilingual_tokenizer.json \
    models/ggml-large-v3.bin; do
    if [ -f "${ROOT_DIR}/${model}" ]; then
        install -m 0644 "${ROOT_DIR}/${model}" "${DIST_DIR}/models/$(basename "${model}")"
    fi
done

if [ "$(uname -s)" = "Linux" ] && command -v ldd >/dev/null 2>&1 && ldd "${DIST_DIR}/linetime" 2>/dev/null | grep -Eq 'libwhisper|libggml'; then
    printf 'Linetime unexpectedly links a Whisper library\n' >&2
    exit 1
fi

if [ "${CUDA_ENABLED}" -eq 1 ]; then
    LD_LIBRARY_PATH="${DIST_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        "${DIST_DIR}/linetime" --help >/dev/null
else
    "${DIST_DIR}/linetime" --help >/dev/null
fi

printf 'Bundle runtime files verified in %s\n' "${DIST_DIR}"
printf 'Required files: linetime, whisper-cli, ffmpeg\n'
if [ "${CUDA_ENABLED}" -eq 1 ]; then
    printf 'GPU libraries: %s\n' "${DIST_DIR}/lib"
fi
if [ "${COREML_ENABLED}" -eq 1 ]; then
    printf 'CoreML provider enabled\n'
fi

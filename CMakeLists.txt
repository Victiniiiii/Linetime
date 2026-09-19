cmake_minimum_required(VERSION 3.14)
project(sounddetect LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Dependencies ----

# FFmpeg (libavcodec, libavformat, libavutil, libswresample)
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVCODEC  REQUIRED libavcodec)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVUTIL  REQUIRED libavutil)
pkg_check_modules(SWRESAMPLE REQUIRED libswresample)

# ONNX Runtime (use local vendor copy)
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/vendor/onnxruntime")
add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include"
)

# Whisper.cpp (built as part of this project)
set(WHISPER_CPP_DIR "${CMAKE_SOURCE_DIR}/vendor/whisper.cpp")
add_subdirectory("${WHISPER_CPP_DIR}" "${CMAKE_BINARY_DIR}/whisper.cpp" EXCLUDE_FROM_ALL)

# ---- sounddetect executable ----

add_executable(sounddetect
    src/main.cpp
    src/audio.cpp
    src/lyrics.cpp
    src/ctc_aligner.cpp
    src/whisper_aligner.cpp
    src/merger.cpp
    src/lrc_writer.cpp
)

target_include_directories(sounddetect PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vendor
    ${AVCODEC_INCLUDE_DIRS}
    ${AVFORMAT_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
    ${SWRESAMPLE_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE}
)

target_link_libraries(sounddetect PRIVATE
    whisper
    ${AVCODEC_LIBRARIES}
    ${AVFORMAT_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWRESAMPLE_LIBRARIES}
    onnxruntime::onnxruntime
)

target_compile_options(sounddetect PRIVATE
    ${AVCODEC_CFLAGS_OTHER}
    ${AVFORMAT_CFLAGS_OTHER}
    ${AVUTIL_CFLAGS_OTHER}
    ${SWRESAMPLE_CFLAGS_OTHER}
)

# Release optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(sounddetect PRIVATE -O3 -DNDEBUG -Wno-unused-result)
endif()

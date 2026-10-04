cmake_minimum_required(VERSION 3.14)
project(linetime VERSION 1.4.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LINETIME_SOURCES
    src/main.cpp
    src/audio.cpp
    src/lyrics.cpp
    src/ctc_aligner.cpp
    src/lrc_writer.cpp
    src/transcriber.cpp
    src/reconcile.cpp
    src/process.cpp
)

set(LINETIME_ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/vendor/onnxruntime" CACHE PATH
    "ONNX Runtime package directory")
option(LINETIME_ONNXRUNTIME_STATIC "Link the static ONNX Runtime archive" OFF)
option(LINETIME_ENABLE_CUDA "Enable the ONNX Runtime CUDA provider" OFF)
option(LINETIME_ENABLE_COREML "Enable the ONNX Runtime CoreML provider" OFF)
option(LINETIME_BUILD_WHISPER_CLI "Build the independent whisper-cli target" ON)
set(LINETIME_WHISPER_DIR "${CMAKE_SOURCE_DIR}/vendor/whisper.cpp" CACHE PATH
    "whisper.cpp source directory")

get_filename_component(LINETIME_ONNXRUNTIME_DIR "${LINETIME_ONNXRUNTIME_DIR}"
                       ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
get_filename_component(LINETIME_WHISPER_DIR "${LINETIME_WHISPER_DIR}"
                       ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")

set(LINETIME_ONNXRUNTIME_INCLUDE_DIRS "")
foreach(candidate
    "${LINETIME_ONNXRUNTIME_DIR}/include"
    "${LINETIME_ONNXRUNTIME_DIR}/include/onnxruntime/core/session")
    if(IS_DIRECTORY "${candidate}")
        list(APPEND LINETIME_ONNXRUNTIME_INCLUDE_DIRS "${candidate}")
    endif()
endforeach()
if(NOT LINETIME_ONNXRUNTIME_INCLUDE_DIRS)
    message(FATAL_ERROR "ONNX Runtime headers were not found in ${LINETIME_ONNXRUNTIME_DIR}")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(ZLIB)

set(LINETIME_ORT_LINK "")
set(LINETIME_ORT_DEFINITIONS "")

if(LINETIME_ONNXRUNTIME_STATIC)
    if(LINETIME_ENABLE_CUDA)
        message(FATAL_ERROR "LINETIME_ENABLE_CUDA requires the shared ONNX Runtime package")
    endif()
    file(GLOB LINETIME_ORT_STATIC_LIBRARIES "${LINETIME_ONNXRUNTIME_DIR}/lib/*.a")
    if(NOT LINETIME_ORT_STATIC_LIBRARIES)
        message(FATAL_ERROR "Static ONNX Runtime libraries were not found in ${LINETIME_ONNXRUNTIME_DIR}/lib")
    endif()
    list(APPEND LINETIME_ORT_LINK "-Wl,--whole-archive")
    list(APPEND LINETIME_ORT_LINK ${LINETIME_ORT_STATIC_LIBRARIES})
    list(APPEND LINETIME_ORT_LINK "-Wl,--no-whole-archive")
    if(UNIX AND NOT APPLE)
        list(INSERT LINETIME_ORT_LINK 0 "-Wl,--allow-multiple-definition")
    endif()
elseif(WIN32)
    find_library(LINETIME_ORT_IMPORT_LIBRARY
        NAMES onnxruntime
        PATHS "${LINETIME_ONNXRUNTIME_DIR}/lib"
        NO_DEFAULT_PATH)
    if(NOT LINETIME_ORT_IMPORT_LIBRARY OR
       NOT EXISTS "${LINETIME_ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
        message(FATAL_ERROR "ONNX Runtime import library and DLL were not found in ${LINETIME_ONNXRUNTIME_DIR}/lib")
    endif()
    add_library(linetime_onnxruntime SHARED IMPORTED)
    set_target_properties(linetime_onnxruntime PROPERTIES
        IMPORTED_IMPLIB "${LINETIME_ORT_IMPORT_LIBRARY}"
        IMPORTED_LOCATION "${LINETIME_ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${LINETIME_ONNXRUNTIME_INCLUDE_DIRS}")
    list(APPEND LINETIME_ORT_LINK linetime_onnxruntime)
    list(APPEND LINETIME_ORT_DEFINITIONS ORT_DLL_IMPORT)
elseif(APPLE)
    find_library(LINETIME_ORT_LIBRARY
        NAMES onnxruntime
        PATHS "${LINETIME_ONNXRUNTIME_DIR}/lib"
        NO_DEFAULT_PATH)
    if(NOT LINETIME_ORT_LIBRARY)
        message(FATAL_ERROR "ONNX Runtime dylib was not found in ${LINETIME_ONNXRUNTIME_DIR}/lib")
    endif()
    add_library(linetime_onnxruntime SHARED IMPORTED)
    set_target_properties(linetime_onnxruntime PROPERTIES
        IMPORTED_LOCATION "${LINETIME_ORT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LINETIME_ONNXRUNTIME_INCLUDE_DIRS}")
    list(APPEND LINETIME_ORT_LINK linetime_onnxruntime)
else()
    find_library(LINETIME_ORT_LIBRARY
        NAMES onnxruntime
        PATHS "${LINETIME_ONNXRUNTIME_DIR}/lib"
        NO_DEFAULT_PATH)
    if(NOT LINETIME_ORT_LIBRARY)
        message(FATAL_ERROR "ONNX Runtime shared library was not found in ${LINETIME_ONNXRUNTIME_DIR}/lib")
    endif()
    add_library(linetime_onnxruntime SHARED IMPORTED)
    set_target_properties(linetime_onnxruntime PROPERTIES
        IMPORTED_LOCATION "${LINETIME_ORT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LINETIME_ONNXRUNTIME_INCLUDE_DIRS}")
    list(APPEND LINETIME_ORT_LINK linetime_onnxruntime)
endif()

if(LINETIME_ENABLE_CUDA)
    if(APPLE OR LINETIME_ONNXRUNTIME_STATIC)
        message(FATAL_ERROR "The CUDA ONNX Runtime provider is supported only with the Linux or Windows shared package")
    endif()
    find_library(LINETIME_ORT_CUDA_PROVIDER
        NAMES onnxruntime_providers_cuda
        PATHS "${LINETIME_ONNXRUNTIME_DIR}/lib" "${LINETIME_ONNXRUNTIME_DIR}"
        NO_DEFAULT_PATH)
    find_library(LINETIME_ORT_SHARED_PROVIDER
        NAMES onnxruntime_providers_shared
        PATHS "${LINETIME_ONNXRUNTIME_DIR}/lib" "${LINETIME_ONNXRUNTIME_DIR}"
        NO_DEFAULT_PATH)
    if(NOT LINETIME_ORT_CUDA_PROVIDER OR NOT LINETIME_ORT_SHARED_PROVIDER)
        message(FATAL_ERROR "CUDA ONNX Runtime provider libraries were not found in ${LINETIME_ONNXRUNTIME_DIR}/lib")
    endif()
    list(APPEND LINETIME_ORT_LINK
        "${LINETIME_ORT_CUDA_PROVIDER}"
        "${LINETIME_ORT_SHARED_PROVIDER}")
    list(APPEND LINETIME_ORT_DEFINITIONS ORT_CUDA)
endif()

set(LINETIME_COREML_LIBRARIES "")
if(LINETIME_ENABLE_COREML)
    if(NOT APPLE)
        message(FATAL_ERROR "LINETIME_ENABLE_COREML is supported only on macOS")
    endif()
    find_library(LINETIME_COREML_FRAMEWORK CoreML)
    find_library(LINETIME_FOUNDATION_FRAMEWORK Foundation)
    if(NOT LINETIME_COREML_FRAMEWORK OR NOT LINETIME_FOUNDATION_FRAMEWORK)
        message(FATAL_ERROR "CoreML and Foundation frameworks were not found")
    endif()
    list(APPEND LINETIME_COREML_LIBRARIES
        "${LINETIME_COREML_FRAMEWORK}"
        "${LINETIME_FOUNDATION_FRAMEWORK}")
    list(APPEND LINETIME_ORT_DEFINITIONS ORT_COREML)
endif()

set(LINETIME_SYSTEM_LIBRARIES Threads::Threads ${CMAKE_DL_LIBS})
list(APPEND LINETIME_SYSTEM_LIBRARIES ${LINETIME_COREML_LIBRARIES})
if(UNIX)
    list(APPEND LINETIME_SYSTEM_LIBRARIES m)
endif()
if(ZLIB_FOUND AND NOT WIN32)
    list(APPEND LINETIME_SYSTEM_LIBRARIES ZLIB::ZLIB)
endif()

add_executable(linetime ${LINETIME_SOURCES})
# MinGW's libstdc++ only declares __gthread_once_t and friends when the
# winpthreads flag is present, so <thread> and <filesystem> fail to parse
# without it. Do not rely on FindThreads to add it for that toolchain.
if(MINGW)
    target_compile_options(linetime PRIVATE -pthread)
    target_link_options(linetime PRIVATE -pthread)
endif()
target_include_directories(linetime PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/vendor"
    ${LINETIME_ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(linetime PRIVATE
    ${LINETIME_ORT_LINK}
    ${LINETIME_SYSTEM_LIBRARIES})
target_compile_definitions(linetime PRIVATE ${LINETIME_ORT_DEFINITIONS})

set_target_properties(linetime PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin"
    SKIP_BUILD_RPATH TRUE)

if(APPLE)
    set_target_properties(linetime PROPERTIES
        INSTALL_RPATH "@loader_path;@loader_path/lib")
    if(NOT LINETIME_ONNXRUNTIME_STATIC)
        target_link_options(linetime PRIVATE
            "-Wl,-rpath,@loader_path"
            "-Wl,-rpath,@loader_path/lib")
    endif()
elseif(UNIX)
    set_target_properties(linetime PROPERTIES
        INSTALL_RPATH "$ORIGIN;$ORIGIN/lib")
    if(NOT LINETIME_ONNXRUNTIME_STATIC)
        target_link_options(linetime PRIVATE
            "-Wl,-rpath,$ORIGIN"
            "-Wl,-rpath,$ORIGIN/lib")
    endif()
endif()

if(LINETIME_BUILD_WHISPER_CLI)
    if(NOT EXISTS "${LINETIME_WHISPER_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "whisper.cpp was not found in ${LINETIME_WHISPER_DIR}")
    endif()
    set(WHISPER_BUILD_EXAMPLES ON CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(WHISPER_COMMON_FFMPEG OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    # whisper.cpp is what actually runs on the accelerator, so it has to follow
    # the requested backend. Forcing this off made a CUDA build silently fall
    # back to CPU while still reporting a GPU provider.
    if(LINETIME_ENABLE_CUDA)
        set(GGML_CUDA ON CACHE BOOL "" FORCE)
    else()
        set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    endif()
    # -march=native bakes the build machine's instruction set into the binary.
    # A release built on an AVX-512 CPU then dies with SIGILL inside
    # ggml_cpu_init on any older CPU, which is called unconditionally at whisper
    # init. Turning native off makes ggml emit runtime-dispatched variants
    # instead, which stays portable and still uses AVX2 where available.
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    if(APPLE)
        if(LINETIME_ENABLE_COREML)
            set(GGML_METAL ON CACHE BOOL "" FORCE)
        else()
            set(GGML_METAL OFF CACHE BOOL "" FORCE)
        endif()
        set(GGML_BLAS OFF CACHE BOOL "" FORCE)
    else()
        set(GGML_METAL OFF CACHE BOOL "" FORCE)
    endif()
    add_subdirectory("${LINETIME_WHISPER_DIR}" "${CMAKE_BINARY_DIR}/whisper" EXCLUDE_FROM_ALL)
    if(NOT TARGET whisper-cli)
        message(FATAL_ERROR "whisper-cli target was not created")
    endif()
    set_target_properties(whisper-cli PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin")
endif()

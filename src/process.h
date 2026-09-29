#pragma once

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
extern "C" char **environ;
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

namespace process {

struct Result {
    bool started = false;
    int exit_code = -1;
    std::string stdout_data;
    std::string stderr_data;
    std::string error;

    bool succeeded() const {
        return started && exit_code == 0 && error.empty();
    }
};

namespace detail {

inline bool has_separator(const std::string& value) {
#ifdef _WIN32
    return value.find('/') != std::string::npos ||
           value.find('\\') != std::string::npos;
#else
    return value.find('/') != std::string::npos;
#endif
}

#ifdef _WIN32

inline std::wstring widen(const std::string& value) {
    if (value.empty()) return std::wstring();
    UINT code = CP_UTF8;
    int size = MultiByteToWideChar(code, MB_ERR_INVALID_CHARS, value.data(),
                                   static_cast<int>(value.size()), nullptr, 0);
    if (size <= 0) {
        code = CP_ACP;
        size = MultiByteToWideChar(code, 0, value.data(),
                                   static_cast<int>(value.size()), nullptr, 0);
    }
    if (size <= 0) return std::wstring();
    std::wstring result(static_cast<size_t>(size), L'\0');
    if (MultiByteToWideChar(code, 0, value.data(), static_cast<int>(value.size()),
                            result.data(), size) <= 0)
        return std::wstring();
    return result;
}

inline std::string narrow(const std::wstring& value) {
    if (value.empty()) return std::string();
    int size = WideCharToMultiByte(CP_UTF8, 0, value.data(),
                                   static_cast<int>(value.size()), nullptr, 0,
                                   nullptr, nullptr);
    if (size <= 0) return std::string();
    std::string result(static_cast<size_t>(size), '\0');
    if (WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
                            result.data(), size, nullptr, nullptr) <= 0)
        return std::string();
    return result;
}

inline std::wstring quote_argument(const std::wstring& value) {
    if (!value.empty() && value.find_first_of(L" \t\n\v\"") == std::wstring::npos)
        return value;

    std::wstring result;
    result.push_back(L'"');
    size_t slashes = 0;
    for (wchar_t c : value) {
        if (c == L'\\') {
            ++slashes;
            continue;
        }
        if (c == L'"') {
            result.append(slashes * 2 + 1, L'\\');
            result.push_back(c);
            slashes = 0;
            continue;
        }
        result.append(slashes, L'\\');
        slashes = 0;
        result.push_back(c);
    }
    result.append(slashes * 2, L'\\');
    result.push_back(L'"');
    return result;
}

inline std::string error_message(DWORD code) {
    LPWSTR text = nullptr;
    DWORD length = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, 0, reinterpret_cast<LPWSTR>(&text), 0, nullptr);
    if (length == 0 || !text) return "Windows error " + std::to_string(code);
    std::wstring message(text, length);
    LocalFree(text);
    while (!message.empty() && (message.back() == L'\r' || message.back() == L'\n'))
        message.pop_back();
    std::string result = narrow(message);
    return result.empty() ? "Windows error " + std::to_string(code) : result;
}

inline void close_handle(HANDLE& handle) {
    if (handle && handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
    handle = nullptr;
}

inline std::wstring merged_environment_block(
    const std::vector<std::string>& additions, bool& valid) {
    valid = true;
    std::map<std::wstring, std::wstring> values;
    std::vector<std::wstring> passthrough;
    auto assign = [&values](const std::wstring& key, const std::wstring& value) {
        for (auto& item : values) {
            if (_wcsicmp(item.first.c_str(), key.c_str()) == 0) {
                item.second = value;
                return;
            }
        }
        values.emplace(key, value);
    };

    LPWCH inherited = GetEnvironmentStringsW();
    if (inherited) {
        const wchar_t* current = inherited;
        while (*current) {
            size_t length = wcslen(current);
            if (length == 0) break;
            std::wstring entry(current, length);
            size_t equals = entry.find(L'=');
            if (equals == std::wstring::npos || equals == 0) {
                passthrough.push_back(entry);
            } else {
                assign(entry.substr(0, equals), entry.substr(equals + 1));
            }
            current += length + 1;
        }
        FreeEnvironmentStringsW(inherited);
    }

    for (const std::string& addition : additions) {
        std::wstring entry = widen(addition);
        size_t equals = entry.find(L'=');
        if (entry.empty() || equals == std::wstring::npos || equals == 0) {
            valid = false;
            continue;
        }
        assign(entry.substr(0, equals), entry.substr(equals + 1));
    }

    std::wstring block;
    for (const std::wstring& entry : passthrough) {
        block += entry;
        block.push_back(L'\0');
    }
    for (const auto& item : values) {
        block += item.first;
        block.push_back(L'=');
        block += item.second;
        block.push_back(L'\0');
    }
    block.push_back(L'\0');
    return block;
}

inline std::string find_on_path(const std::string& name,
                                const std::vector<std::string>& environment) {
    if (has_separator(name)) return name;
    std::wstring path;
    for (const std::string& entry : environment) {
        if (entry.size() >= 5 && _wcsicmp(widen(entry.substr(0, 5)).c_str(), L"PATH=") == 0) {
            path = widen(entry.substr(5));
            break;
        }
    }
    if (path.empty()) path = L".;";
    std::wstring wide_name = widen(name);
    size_t start = 0;
    while (start <= path.size()) {
        size_t end = path.find(L';', start);
        if (end == std::wstring::npos) end = path.size();
        std::wstring directory = path.substr(start, end - start);
        if (directory.empty()) directory = L".";
        std::wstring candidate = directory;
        if (!candidate.empty() && candidate.back() != L'\\' && candidate.back() != L'/')
            candidate.push_back(L'\\');
        candidate += wide_name;
        DWORD attributes = GetFileAttributesW(candidate.c_str());
        if (attributes != INVALID_FILE_ATTRIBUTES &&
            !(attributes & FILE_ATTRIBUTE_DIRECTORY))
            return narrow(candidate);
        if (wide_name.find(L'.') == std::wstring::npos) {
            candidate += L".exe";
            attributes = GetFileAttributesW(candidate.c_str());
            if (attributes != INVALID_FILE_ATTRIBUTES &&
                !(attributes & FILE_ATTRIBUTE_DIRECTORY))
                return narrow(candidate);
        }
        if (end == path.size()) break;
        start = end + 1;
    }
    return std::string();
}

inline std::string find_on_path_block(const std::string& name,
                                      const std::wstring& environment) {
    if (has_separator(name)) return name;
    std::wstring path;
    const wchar_t* current = environment.c_str();
    while (current && *current) {
        size_t length = wcslen(current);
        if (length == 0) break;
        std::wstring entry(current, length);
        if (entry.size() >= 5 && _wcsicmp(entry.substr(0, 5).c_str(), L"PATH=") == 0) {
            path = entry.substr(5);
            break;
        }
        current += length + 1;
    }
    if (path.empty()) path = L".;";
    std::wstring wide_name = widen(name);
    size_t start = 0;
    while (start <= path.size()) {
        size_t end = path.find(L';', start);
        if (end == std::wstring::npos) end = path.size();
        std::wstring directory = path.substr(start, end - start);
        if (directory.empty()) directory = L".";
        std::wstring candidate = directory;
        if (!candidate.empty() && candidate.back() != L'\\' && candidate.back() != L'/')
            candidate.push_back(L'\\');
        candidate += wide_name;
        DWORD attributes = GetFileAttributesW(candidate.c_str());
        if (attributes != INVALID_FILE_ATTRIBUTES &&
            !(attributes & FILE_ATTRIBUTE_DIRECTORY))
            return narrow(candidate);
        if (wide_name.find(L'.') == std::wstring::npos) {
            candidate += L".exe";
            attributes = GetFileAttributesW(candidate.c_str());
            if (attributes != INVALID_FILE_ATTRIBUTES &&
                !(attributes & FILE_ATTRIBUTE_DIRECTORY))
                return narrow(candidate);
        }
        if (end == path.size()) break;
        start = end + 1;
    }
    return std::string();
}

inline void read_pipe(HANDLE handle, std::string& output) {
    char buffer[8192];
    for (;;) {
        DWORD count = 0;
        if (!ReadFile(handle, buffer, sizeof(buffer), &count, nullptr)) {
            if (GetLastError() == ERROR_BROKEN_PIPE) break;
            break;
        }
        if (count == 0) break;
        output.append(buffer, buffer + count);
    }
}

#else

inline std::vector<std::string> merged_environment(
    const std::vector<std::string>& additions, bool& valid) {
    valid = true;
    std::vector<std::string> values;
    for (char** current = environ; current && *current; ++current)
        values.emplace_back(*current);

    for (const std::string& addition : additions) {
        size_t equals = addition.find('=');
        if (addition.empty() || equals == std::string::npos || equals == 0) {
            valid = false;
            continue;
        }
        std::string key = addition.substr(0, equals);
        bool replaced = false;
        for (std::string& value : values) {
            size_t current_equals = value.find('=');
            if (current_equals != std::string::npos && value.substr(0, current_equals) == key) {
                value = addition;
                replaced = true;
                break;
            }
        }
        if (!replaced) values.push_back(addition);
    }
    return values;
}

inline std::string find_on_path(const std::string& name,
                                const std::vector<std::string>& environment) {
    if (has_separator(name)) return name;
    std::string path;
    for (const std::string& entry : environment) {
        if (entry.size() >= 5 && entry.compare(0, 5, "PATH=") == 0) {
            path = entry.substr(5);
            break;
        }
    }
    if (path.empty()) path = "/usr/local/bin:/usr/bin:/bin";

    size_t start = 0;
    while (start <= path.size()) {
        size_t end = path.find(':', start);
        if (end == std::string::npos) end = path.size();
        std::string directory = path.substr(start, end - start);
        if (directory.empty()) directory = ".";
        std::string candidate = directory;
        if (candidate.back() != '/') candidate.push_back('/');
        candidate += name;
        if (access(candidate.c_str(), X_OK) == 0) return candidate;
        if (end == path.size()) break;
        start = end + 1;
    }
    return std::string();
}

inline void close_fd(int& fd) {
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
}

#endif

}

inline Result run(const std::vector<std::string>& argv,
                  const std::vector<std::string>& environment_additions = {}) {
    Result result;
    if (argv.empty() || argv[0].empty()) {
        result.error = "process executable is empty";
        return result;
    }

#ifdef _WIN32
    bool environment_valid = false;
    std::wstring environment_block =
        detail::merged_environment_block(environment_additions, environment_valid);
    if (!environment_valid) {
        result.error = "invalid environment addition";
        return result;
    }

    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;

    HANDLE stdout_read = nullptr;
    HANDLE stdout_write = nullptr;
    HANDLE stderr_read = nullptr;
    HANDLE stderr_write = nullptr;
    HANDLE null_input = nullptr;
    if (!CreatePipe(&stdout_read, &stdout_write, &security, 0) ||
        !CreatePipe(&stderr_read, &stderr_write, &security, 0)) {
        result.error = "failed to create process pipes";
        detail::close_handle(stdout_read);
        detail::close_handle(stdout_write);
        detail::close_handle(stderr_read);
        detail::close_handle(stderr_write);
        return result;
    }
    if (!SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0) ||
        !SetHandleInformation(stderr_read, HANDLE_FLAG_INHERIT, 0)) {
        result.error = "failed to configure process pipes";
        detail::close_handle(stdout_read);
        detail::close_handle(stdout_write);
        detail::close_handle(stderr_read);
        detail::close_handle(stderr_write);
        return result;
    }
    null_input = CreateFileW(L"NUL", GENERIC_READ,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, &security,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (!null_input) {
        result.error = "failed to open process input";
        detail::close_handle(stdout_read);
        detail::close_handle(stdout_write);
        detail::close_handle(stderr_read);
        detail::close_handle(stderr_write);
        return result;
    }

    std::wstring command_line;
    for (size_t i = 0; i < argv.size(); ++i) {
        if (i) command_line.push_back(L' ');
        command_line += detail::quote_argument(detail::widen(argv[i]));
    }
    std::vector<wchar_t> mutable_command(command_line.begin(), command_line.end());
    mutable_command.push_back(L'\0');

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = null_input;
    startup.hStdOutput = stdout_write;
    startup.hStdError = stderr_write;
    PROCESS_INFORMATION process_info{};

    std::string resolved_executable = detail::find_on_path_block(argv[0], environment_block);
    std::wstring application_name = detail::widen(resolved_executable);
    BOOL created = CreateProcessW(
        application_name.empty() ? nullptr : application_name.c_str(),
        mutable_command.data(), nullptr, nullptr, TRUE,
        CREATE_UNICODE_ENVIRONMENT,
        environment_block.empty() ? nullptr :
            const_cast<wchar_t*>(environment_block.data()),
        nullptr, &startup, &process_info);
    DWORD create_error = created ? ERROR_SUCCESS : GetLastError();
    detail::close_handle(null_input);
    detail::close_handle(stdout_write);
    detail::close_handle(stderr_write);
    if (!created) {
        result.error = "failed to start process: " + detail::error_message(create_error);
        detail::close_handle(stdout_read);
        detail::close_handle(stderr_read);
        return result;
    }

    result.started = true;
    std::thread stdout_thread;
    std::thread stderr_thread;
    try {
        stdout_thread = std::thread(detail::read_pipe, stdout_read, std::ref(result.stdout_data));
        stderr_thread = std::thread(detail::read_pipe, stderr_read, std::ref(result.stderr_data));
    } catch (const std::exception& error) {
        TerminateProcess(process_info.hProcess, 1);
        WaitForSingleObject(process_info.hProcess, INFINITE);
        if (stdout_thread.joinable()) stdout_thread.join();
        detail::close_handle(stdout_read);
        detail::close_handle(stderr_read);
        CloseHandle(process_info.hThread);
        CloseHandle(process_info.hProcess);
        result.error = "failed to read process output: " + std::string(error.what());
        result.started = false;
        return result;
    }

    DWORD wait_result = WaitForSingleObject(process_info.hProcess, INFINITE);
    if (wait_result == WAIT_FAILED) {
        result.error = "failed to wait for process: " +
                       detail::error_message(GetLastError());
        TerminateProcess(process_info.hProcess, 1);
        WaitForSingleObject(process_info.hProcess, INFINITE);
    }
    if (stdout_thread.joinable()) stdout_thread.join();
    if (stderr_thread.joinable()) stderr_thread.join();
    DWORD exit_code = 0;
    if (wait_result != WAIT_FAILED && GetExitCodeProcess(process_info.hProcess, &exit_code)) {
        result.exit_code = exit_code > static_cast<DWORD>(INT_MAX)
                               ? -1
                               : static_cast<int>(exit_code);
        if (exit_code > static_cast<DWORD>(INT_MAX))
            result.error = "process exit code is out of range";
    } else if (result.error.empty()) {
        result.error = "failed to read process exit code";
    }
    detail::close_handle(stdout_read);
    detail::close_handle(stderr_read);
    CloseHandle(process_info.hThread);
    CloseHandle(process_info.hProcess);
    return result;
#else
    bool environment_valid = false;
    std::vector<std::string> environment =
        detail::merged_environment(environment_additions, environment_valid);
    if (!environment_valid) {
        result.error = "invalid environment addition";
        return result;
    }

    int stdout_pipe[2] = {-1, -1};
    int stderr_pipe[2] = {-1, -1};
    int exec_error_pipe[2] = {-1, -1};
    if (pipe(stdout_pipe) != 0) {
        result.error = std::string("failed to create stdout pipe: ") + std::strerror(errno);
        return result;
    }
    if (pipe(stderr_pipe) != 0) {
        result.error = std::string("failed to create stderr pipe: ") + std::strerror(errno);
        detail::close_fd(stdout_pipe[0]);
        detail::close_fd(stdout_pipe[1]);
        return result;
    }
    if (pipe(exec_error_pipe) != 0) {
        result.error = std::string("failed to create process status pipe: ") + std::strerror(errno);
        detail::close_fd(stdout_pipe[0]);
        detail::close_fd(stdout_pipe[1]);
        detail::close_fd(stderr_pipe[0]);
        detail::close_fd(stderr_pipe[1]);
        return result;
    }

    auto set_close_on_exec = [](int fd) {
        int flags = fcntl(fd, F_GETFD, 0);
        if (flags >= 0) fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
    };
    set_close_on_exec(stdout_pipe[0]);
    set_close_on_exec(stderr_pipe[0]);
    set_close_on_exec(exec_error_pipe[0]);
    set_close_on_exec(exec_error_pipe[1]);

    std::string executable = detail::find_on_path(argv[0], environment);
    if (executable.empty()) executable = argv[0];
    std::vector<char*> child_argv;
    child_argv.reserve(argv.size() + 1);
    for (const std::string& value : argv)
        child_argv.push_back(const_cast<char*>(value.c_str()));
    child_argv.push_back(nullptr);
    std::vector<char*> child_environment;
    child_environment.reserve(environment.size() + 1);
    for (const std::string& value : environment)
        child_environment.push_back(const_cast<char*>(value.c_str()));
    child_environment.push_back(nullptr);

    pid_t child = fork();
    if (child < 0) {
        result.error = std::string("failed to start process: ") + std::strerror(errno);
        detail::close_fd(stdout_pipe[0]);
        detail::close_fd(stdout_pipe[1]);
        detail::close_fd(stderr_pipe[0]);
        detail::close_fd(stderr_pipe[1]);
        detail::close_fd(exec_error_pipe[0]);
        detail::close_fd(exec_error_pipe[1]);
        return result;
    }

    if (child == 0) {
        detail::close_fd(stdout_pipe[0]);
        detail::close_fd(stderr_pipe[0]);
        detail::close_fd(exec_error_pipe[0]);
        if (dup2(stdout_pipe[1], STDOUT_FILENO) < 0 ||
            dup2(stderr_pipe[1], STDERR_FILENO) < 0) {
            const char message[] = "process setup failed\n";
            ssize_t ignored = write(exec_error_pipe[1], message, sizeof(message) - 1);
            (void)ignored;
            _exit(127);
        }
        detail::close_fd(stdout_pipe[1]);
        detail::close_fd(stderr_pipe[1]);
        execve(executable.c_str(), child_argv.data(), child_environment.data());
        const char message[] = "failed to execute process\n";
        ssize_t ignored = write(exec_error_pipe[1], message, sizeof(message) - 1);
        (void)ignored;
        _exit(127);
    }

    result.started = true;
    detail::close_fd(stdout_pipe[1]);
    detail::close_fd(stderr_pipe[1]);
    detail::close_fd(exec_error_pipe[1]);

    int stdout_fd = stdout_pipe[0];
    int stderr_fd = stderr_pipe[0];
    auto set_nonblocking = [](int fd) {
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    };
    set_nonblocking(stdout_fd);
    set_nonblocking(stderr_fd);

    while (stdout_fd >= 0 || stderr_fd >= 0) {
        pollfd descriptors[2]{};
        descriptors[0].fd = stdout_fd;
        descriptors[0].events = POLLIN;
        descriptors[1].fd = stderr_fd;
        descriptors[1].events = POLLIN;
        nfds_t count = 0;
        if (stdout_fd >= 0) ++count;
        if (stderr_fd >= 0) ++count;
        int poll_result = poll(descriptors, count, -1);
        if (poll_result < 0) {
            if (errno == EINTR) continue;
            result.error = std::string("failed to poll process output: ") + std::strerror(errno);
            detail::close_fd(stdout_fd);
            detail::close_fd(stderr_fd);
            break;
        }
        for (int i = 0; i < 2; ++i) {
            int& fd = i == 0 ? stdout_fd : stderr_fd;
            if (fd < 0 || descriptors[i].revents == 0) continue;
            if (descriptors[i].revents & (POLLNVAL | POLLERR)) {
                detail::close_fd(fd);
                continue;
            }
            std::string& destination = i == 0 ? result.stdout_data : result.stderr_data;
            for (;;) {
                char buffer[8192];
                ssize_t bytes = read(fd, buffer, sizeof(buffer));
                if (bytes > 0) {
                    destination.append(buffer, buffer + bytes);
                    continue;
                }
                if (bytes < 0 && errno == EINTR) continue;
                if (bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
                detail::close_fd(fd);
                break;
            }
        }
    }

    int status = 0;
    pid_t waited;
    int wait_error = 0;
    do {
        waited = waitpid(child, &status, 0);
        if (waited < 0 && errno != EINTR) wait_error = errno;
    } while (waited < 0 && errno == EINTR);
    char error_buffer[256];
    ssize_t error_size = read(exec_error_pipe[0], error_buffer, sizeof(error_buffer) - 1);
    if (error_size > 0) {
        error_buffer[error_size] = '\0';
        result.error = error_buffer;
    }
    detail::close_fd(exec_error_pipe[0]);
    detail::close_fd(stdout_fd);
    detail::close_fd(stderr_fd);

    if (waited < 0) {
        if (result.error.empty())
            result.error = std::string("failed to wait for process: ") + std::strerror(wait_error);
    } else if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.exit_code = 128 + WTERMSIG(status);
        if (result.error.empty())
            result.error = "process terminated by signal " + std::to_string(WTERMSIG(status));
    } else {
        result.error = "process ended without an exit status";
    }
    return result;
#endif
}

inline unsigned long long current_process_id() {
#ifdef _WIN32
    return static_cast<unsigned long long>(GetCurrentProcessId());
#else
    return static_cast<unsigned long long>(getpid());
#endif
}

inline std::string executable_path(const char* argv0 = nullptr) {
    std::string result;
#ifdef _WIN32
    std::vector<wchar_t> buffer(512);
    for (;;) {
        DWORD length = GetModuleFileNameW(nullptr, buffer.data(),
                                          static_cast<DWORD>(buffer.size()));
        if (length == 0) break;
        if (length < buffer.size()) {
            result = detail::narrow(std::wstring(buffer.data(), length));
            break;
        }
        buffer.resize(buffer.size() * 2);
    }
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    if (size > 0) {
        std::vector<char> buffer(size + 1);
        if (_NSGetExecutablePath(buffer.data(), &size) == 0) {
            result.assign(buffer.data());
            std::error_code error;
            std::filesystem::path canonical = std::filesystem::weakly_canonical(
                std::filesystem::path(result), error);
            if (!error) result = canonical.string();
        }
    }
#else
    std::vector<char> buffer(512);
    for (;;) {
        ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size());
        if (length < 0) break;
        if (static_cast<size_t>(length) < buffer.size()) {
            result.assign(buffer.data(), static_cast<size_t>(length));
            break;
        }
        buffer.resize(buffer.size() * 2);
    }
#endif
    if (!result.empty()) return result;
    if (argv0 && *argv0) {
#ifdef _WIN32
        std::wstring wide_argv0 = detail::widen(argv0);
        std::wstring search_path;
        if (GetEnvironmentVariableW(L"PATH", nullptr, 0) == 0) {
            DWORD length = GetEnvironmentVariableW(L"PATH", nullptr, 0);
            search_path.resize(length);
            if (length > 0)
                GetEnvironmentVariableW(L"PATH", search_path.data(), length);
        } else {
            DWORD length = GetEnvironmentVariableW(L"PATH", nullptr, 0);
            search_path.resize(length);
            if (length > 0)
                GetEnvironmentVariableW(L"PATH", search_path.data(), length);
        }
        if (!wide_argv0.empty() && !search_path.empty()) {
            std::wstring current;
            size_t start = 0;
            while (start <= search_path.size()) {
                size_t end = search_path.find(L';', start);
                if (end == std::wstring::npos) end = search_path.size();
                std::wstring directory = search_path.substr(start, end - start);
                if (directory.empty()) directory = L".";
                std::wstring candidate = directory;
                if (!candidate.empty() && candidate.back() != L'\\' && candidate.back() != L'/')
                    candidate.push_back(L'\\');
                candidate += wide_argv0;
                DWORD attributes = GetFileAttributesW(candidate.c_str());
                if (attributes != INVALID_FILE_ATTRIBUTES &&
                    !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    result = detail::narrow(candidate);
                    break;
                }
                if (end == search_path.size()) break;
                start = end + 1;
            }
        }
#else
        std::vector<std::string> environment;
        for (char** current = environ; current && *current; ++current)
            environment.emplace_back(*current);
        std::string resolved = detail::find_on_path(argv0, environment);
        if (!resolved.empty()) result = resolved;
#endif
        if (result.empty()) {
            std::error_code error;
            std::filesystem::path fallback = std::filesystem::absolute(
                std::filesystem::path(argv0), error);
            if (!error) result = fallback.string();
        }
    }
    return result;
}

inline std::string executable_directory(const char* argv0 = nullptr) {
    std::string path = executable_path(argv0);
    if (path.empty()) return ".";
    std::error_code error;
    std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (parent.empty()) return ".";
    std::filesystem::path normalized = std::filesystem::weakly_canonical(parent, error);
    if (!error) return normalized.string();
    return parent.string();
}

}

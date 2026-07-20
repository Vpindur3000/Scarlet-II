#pragma once

#include <filesystem>
#include <string_view>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Scarlet::RuntimePaths {

inline std::filesystem::path executable_dir() {
#if defined(_WIN32)
    std::wstring buffer(32768, L'\0');
    const DWORD size = ::GetModuleFileNameW(nullptr, buffer.data(), DWORD(buffer.size()));
    if (size > 0 && size < buffer.size()) {
        buffer.resize(size);
        return std::filesystem::path(buffer).parent_path();
    }
#elif defined(__linux__)
    std::error_code ec;
    const auto exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) return exe.parent_path();
#endif
    return std::filesystem::current_path();
}

inline std::filesystem::path explicit_path(std::string_view value) {
    std::filesystem::path path{std::string(value)};
    if (path.is_relative()) path = std::filesystem::current_path() / path;
    return path.lexically_normal();
}

inline std::filesystem::path default_asset(std::string_view relative) {
    return (executable_dir() / std::filesystem::path(std::string(relative))).lexically_normal();
}

} // namespace Scarlet::RuntimePaths

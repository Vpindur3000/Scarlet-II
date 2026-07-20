cmake_minimum_required(VERSION 3.20)
project(ScarletII_ModernBStar VERSION 1.0.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SCARLET_ZEN3_AVX2 "Tune for AMD Zen 3 with AVX2/BMI/FMA/POPCNT" ON)
option(SCARLET_USE_ZLIB "Enable gzip LD2 weights through zlib when available" ON)
option(SCARLET_USE_OPENMP "Parallelize LD2 raw inference with OpenMP when available" ON)

if(SCARLET_USE_ZLIB)
    find_package(ZLIB QUIET)
endif()
if(SCARLET_USE_OPENMP)
    find_package(OpenMP QUIET)
endif()

add_library(scarlet_core STATIC
    src/movegen.cpp
    src/eval.cpp
    src/nnue_berserk.cpp
    src/leela_raw.cpp
    src/syzygy.cpp
    src/search.cpp
    src/uci.cpp
    third_party/pyrrhic/tbprobe.c
)

set_source_files_properties(third_party/pyrrhic/tbprobe.c PROPERTIES LANGUAGE CXX)

target_include_directories(scarlet_core PUBLIC src PRIVATE third_party/pyrrhic)
target_compile_definitions(scarlet_core PRIVATE SCARLET_PYRRHIC=1)
if(SCARLET_USE_ZLIB AND ZLIB_FOUND)
    target_compile_definitions(scarlet_core PRIVATE SCARLET_USE_ZLIB=1)
    target_link_libraries(scarlet_core PUBLIC ZLIB::ZLIB)
else()
    message(STATUS "Building without zlib gzip support; use an uncompressed network file")
endif()

if(SCARLET_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_compile_definitions(scarlet_core PRIVATE SCARLET_USE_OPENMP=1)
    target_link_libraries(scarlet_core PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_options(scarlet_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wshadow>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

if(SCARLET_ZEN3_AVX2)
    target_compile_options(scarlet_core PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-march=znver3;-mtune=znver3;-mavx2;-mbmi;-mbmi2;-mpopcnt;-mfma>
        $<$<CXX_COMPILER_ID:MSVC>:/O2;/arch:AVX2>
    )
else()
    target_compile_options(scarlet_core PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-mavx2;-mbmi;-mbmi2;-mpopcnt;-mfma>
        $<$<CXX_COMPILER_ID:MSVC>:/O2;/arch:AVX2>
    )
endif()

add_executable(scarlet src/main.cpp)
target_link_libraries(scarlet PRIVATE scarlet_core)

add_executable(scarlet-perft src/perft.cpp)
target_link_libraries(scarlet-perft PRIVATE scarlet_core)

if(BUILD_TESTING)
    add_executable(scarlet-core-tests tests/core_tests.cpp)
    target_link_libraries(scarlet-core-tests PRIVATE scarlet_core)
    target_include_directories(scarlet-core-tests PRIVATE third_party/lc0-policy-map)
    add_test(NAME scarlet-core-tests COMMAND scarlet-core-tests)
    if(WIN32)
        find_program(SCARLET_POWERSHELL_EXECUTABLE powershell REQUIRED)
        add_test(NAME scarlet-uci-async
            COMMAND ${SCARLET_POWERSHELL_EXECUTABLE} -NoProfile -ExecutionPolicy Bypass
                    -File ${CMAKE_CURRENT_SOURCE_DIR}/tests/uci_async.ps1
                    -Engine $<TARGET_FILE:scarlet>)
    endif()
endif()

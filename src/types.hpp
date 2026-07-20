#pragma once

#include "movegen.hpp"
#include <cstdint>
#include <limits>
#include <string>

namespace Scarlet {

using Value = int;
using Depth = int;
using CompactMove = TTMove;

inline constexpr Value VALUE_ZERO = 0;
inline constexpr Value VALUE_DRAW = 0;
inline constexpr Value VALUE_NONE = 32767;
inline constexpr Value VALUE_INFINITE = 32001;
inline constexpr Value VALUE_MATE = 32000;
inline constexpr int MAX_PLY = 246;
inline constexpr Value VALUE_MATE_IN_MAX_PLY = VALUE_MATE - MAX_PLY;
inline constexpr Value VALUE_MATED_IN_MAX_PLY = -VALUE_MATE_IN_MAX_PLY;

inline constexpr const char* START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

inline bool is_mate_score(Value v) {
    return v >= VALUE_MATE_IN_MAX_PLY || v <= VALUE_MATED_IN_MAX_PLY;
}

inline Value mate_in(int ply) { return VALUE_MATE - ply; }
inline Value mated_in(int ply) { return -VALUE_MATE + ply; }

} // namespace Scarlet

#pragma once

#include "types.hpp"
#include <string>

namespace Scarlet::NNUE { struct Accumulator; }

namespace Scarlet::Eval {

Value evaluate_hce_white(const Position& pos);
Value evaluate_hce(const Position& pos);

// Primary eval slot. Uses Berserk 14 NNUE when the backend is loaded/enabled;
// otherwise safely falls back to the old Scarlet HCE.
Value evaluate(const Position& pos, const NNUE::Accumulator* accumulator = nullptr);
Value evaluate_white(const Position& pos);

bool init_backends();
bool load_berserk_nnue(const std::string& path);
void set_berserk_nnue_enabled(bool enabled);
std::string backend_status();

} // namespace Scarlet::Eval

#include "syzygy.hpp"

#include "tbprobe.h"

#include <algorithm>
#include <array>

namespace Scarlet::Syzygy {

namespace {

constexpr unsigned no_ep(const Position& pos) {
    return pos.epSquare >= 0 ? unsigned(pos.epSquare) : 0u;
}

unsigned pyrrhic_promotion(PieceType type) {
    switch (type) {
        case QUEEN:  return 1;
        case ROOK:   return 2;
        case BISHOP: return 3;
        case KNIGHT: return 4;
        default:     return 0;
    }
}

} // namespace

Backend& Backend::instance() {
    static Backend backend;
    return backend;
}

Backend::~Backend() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    tb_free();
}

void Backend::set_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    status_.enabled = enabled;
}

void Backend::set_probe_limit(int pieces) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    // Scarlet intentionally caps use at six men. Pyrrhic can index seven-men
    // files, but that is outside this release's memory/performance contract.
    status_.probe_limit = std::clamp(pieces, 0, 6);
}

bool Backend::set_path(std::string_view path) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    const std::string value(path);
    const bool ok = tb_init(value.c_str());
    status_.path = value;
    status_.largest = TB_LARGEST;
    status_.loaded = ok && TB_LARGEST > 0;
    if (!ok) {
        ++status_.failures;
        status_.message = "Syzygy initialization failed";
    } else if (TB_LARGEST <= 0) {
        status_.message = "Syzygy path contains no recognized WDL tablebases";
    } else {
        status_.message = "Syzygy WDL/DTZ tablebases loaded";
    }
    return status_.loaded;
}

bool Backend::eligible(const Position& pos) const {
    return status_.enabled && status_.loaded && status_.probe_limit > 0
        && pos.castlingRights == 0
        && popcount(pos.occAll) <= std::min(status_.probe_limit, status_.largest);
}

std::optional<Wdl> Backend::decode_wdl(unsigned raw) {
    switch (raw) {
        case TB_LOSS:         return Wdl::Loss;
        case TB_BLESSED_LOSS: return Wdl::BlessedLoss;
        case TB_DRAW:         return Wdl::Draw;
        case TB_CURSED_WIN:   return Wdl::CursedWin;
        case TB_WIN:          return Wdl::Win;
        default:              return std::nullopt;
    }
}

Move Backend::resolve_move(const Position& pos, unsigned raw) {
    const unsigned from = TB_GET_FROM(raw);
    const unsigned to = TB_GET_TO(raw);
    const unsigned promotion = TB_GET_PROMOTES(raw);
    MoveList legal;
    Position copy = pos;
    generate_legal(copy, legal);
    for (Move move : legal) {
        if (unsigned(from_sq(move)) != from || unsigned(to_sq(move)) != to) continue;
        if (pyrrhic_promotion(promotion_type(move)) == promotion) return move;
    }
    return MOVE_NONE;
}

std::optional<Wdl> Backend::probe_wdl(const Position& pos) {
    if (!eligible(pos)) {
        if (pos.castlingRights != 0) ++status_.skipped_castling;
        return std::nullopt;
    }
    // WDL has no current-rule50 input. A DTZ root probe handles non-zero
    // clocks exactly; suppressing this interior shortcut is conservative and
    // avoids a false exact bound near a 50-move claim.
    if (pos.halfmoveClock != 0) {
        ++status_.skipped_rule50;
        return std::nullopt;
    }

    ++status_.wdl_probes;
    const unsigned raw = tb_probe_wdl(
        pos.occ[WHITE], pos.occ[BLACK],
        pos.pieces[WHITE][KING] | pos.pieces[BLACK][KING],
        pos.pieces[WHITE][QUEEN] | pos.pieces[BLACK][QUEEN],
        pos.pieces[WHITE][ROOK] | pos.pieces[BLACK][ROOK],
        pos.pieces[WHITE][BISHOP] | pos.pieces[BLACK][BISHOP],
        pos.pieces[WHITE][KNIGHT] | pos.pieces[BLACK][KNIGHT],
        pos.pieces[WHITE][PAWN] | pos.pieces[BLACK][PAWN],
        no_ep(pos), pos.side == WHITE);
    if (raw == TB_RESULT_FAILED) {
        ++status_.failures;
        return std::nullopt;
    }
    if (auto wdl = decode_wdl(raw)) {
        ++status_.wdl_hits;
        return wdl;
    }
    ++status_.failures;
    return std::nullopt;
}

std::optional<RootProbe> Backend::probe_root(const Position& pos) {
    if (!eligible(pos)) {
        if (pos.castlingRights != 0) ++status_.skipped_castling;
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(root_mutex_);
    ++status_.root_probes;
    const unsigned raw = tb_probe_root(
        pos.occ[WHITE], pos.occ[BLACK],
        pos.pieces[WHITE][KING] | pos.pieces[BLACK][KING],
        pos.pieces[WHITE][QUEEN] | pos.pieces[BLACK][QUEEN],
        pos.pieces[WHITE][ROOK] | pos.pieces[BLACK][ROOK],
        pos.pieces[WHITE][BISHOP] | pos.pieces[BLACK][BISHOP],
        pos.pieces[WHITE][KNIGHT] | pos.pieces[BLACK][KNIGHT],
        pos.pieces[WHITE][PAWN] | pos.pieces[BLACK][PAWN],
        unsigned(std::clamp(pos.halfmoveClock, 0, 100)), no_ep(pos), pos.side == WHITE, nullptr);
    if (raw == TB_RESULT_FAILED || raw == TB_RESULT_CHECKMATE || raw == TB_RESULT_STALEMATE) {
        if (raw == TB_RESULT_FAILED) ++status_.failures;
        return std::nullopt;
    }
    const auto wdl = decode_wdl(TB_GET_WDL(raw));
    const Move move = wdl ? resolve_move(pos, raw) : MOVE_NONE;
    if (!wdl || move == MOVE_NONE) {
        ++status_.failures;
        return std::nullopt;
    }
    ++status_.root_hits;
    return RootProbe{move, *wdl, TB_GET_DTZ(raw)};
}

} // namespace Scarlet::Syzygy

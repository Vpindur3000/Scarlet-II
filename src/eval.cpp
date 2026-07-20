#include "eval.hpp"
#include "nnue_berserk.hpp"
#include "leela_raw.hpp"
#include "syzygy.hpp"

#include <array>
#include <cstdlib>
#include <cstdint>

namespace Scarlet::Eval {

namespace {


struct EvalCacheEntry {
    Key key = 0;
    Color side = WHITE;
    std::uint32_t generation = 0;
    Value score = VALUE_NONE;
};

constexpr std::size_t EVAL_CACHE_SIZE = 1u << 16;
std::array<EvalCacheEntry, EVAL_CACHE_SIZE> EvalCache{};
std::uint32_t EvalCacheGeneration = 1;
std::uint64_t EvalCacheHits = 0;

void clear_eval_cache() {
    ++EvalCacheGeneration;
    if (EvalCacheGeneration == 0) {
        EvalCacheGeneration = 1;
        for (auto& e : EvalCache) e.generation = 0;
    }
}

constexpr std::array<int, PIECE_TYPE_NB> PieceValue = {
    0, 100, 320, 330, 500, 900, 0
};

constexpr std::array<int, 64> PawnPsqt = {
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0
};

constexpr std::array<int, 64> KnightPsqt = {
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
};

constexpr std::array<int, 64> BishopPsqt = {
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
};

constexpr std::array<int, 64> RookPsqt = {
      0,   0,   5,  10,  10,   5,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   5,   5,   0,   0,   0
};

constexpr std::array<int, 64> QueenPsqt = {
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
      0,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20
};

constexpr std::array<int, 64> KingPsqt = {
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30
};

int mirror(int sq) {
    return sq ^ 56;
}

int psqt(PieceType pt, int sq, Color c) {
    const int idx = c == WHITE ? sq : mirror(sq);
    switch (pt) {
        case PAWN:   return PawnPsqt[idx];
        case KNIGHT: return KnightPsqt[idx];
        case BISHOP: return BishopPsqt[idx];
        case ROOK:   return RookPsqt[idx];
        case QUEEN:  return QueenPsqt[idx];
        case KING:   return KingPsqt[idx];
        default:     return 0;
    }
}

int mobility_bonus(const Position& pos, Color c) {
    const Bitboard own = pos.occ[c];
    const Bitboard occ = pos.occAll;
    int bonus = 0;

    Bitboard b = pos.pieces[c][KNIGHT];
    while (b) bonus += 4 * popcount(Attacks::Knight[pop_lsb(b)] & ~own);

    b = pos.pieces[c][BISHOP];
    while (b) bonus += 3 * popcount(Attacks::bishop_attacks(pop_lsb(b), occ) & ~own);

    b = pos.pieces[c][ROOK];
    while (b) bonus += 2 * popcount(Attacks::rook_attacks(pop_lsb(b), occ) & ~own);

    b = pos.pieces[c][QUEEN];
    while (b) bonus += popcount(Attacks::queen_attacks(pop_lsb(b), occ) & ~own);

    return bonus;
}

int passed_pawn_bonus(const Position& pos, Color c) {
    int bonus = 0;
    Bitboard pawns = pos.pieces[c][PAWN];
    const Bitboard enemyPawns = pos.pieces[~c][PAWN];
    while (pawns) {
        const int s = pop_lsb(pawns);
        const int f = file_of(s);
        const int r = rank_of(s);
        Bitboard mask = 0;
        for (int df = -1; df <= 1; ++df) {
            const int nf = f + df;
            if (nf < 0 || nf > 7) continue;
            if (c == WHITE) {
                for (int nr = r + 1; nr < 8; ++nr) mask |= sq_bb(nr * 8 + nf);
            } else {
                for (int nr = r - 1; nr >= 0; --nr) mask |= sq_bb(nr * 8 + nf);
            }
        }
        if ((mask & enemyPawns) == 0) {
            const int advance = c == WHITE ? r : 7 - r;
            bonus += 10 + advance * advance * 3;
        }
    }
    return bonus;
}

} // namespace

Value evaluate_hce_white(const Position& pos) {
    int score = 0;

    for (int s = 0; s < 64; ++s) {
        const Piece pc = pos.board[s];
        if (pc == EMPTY) continue;
        const Color c = color_of(pc);
        const PieceType pt = type_of(pc);
        const int v = PieceValue[pt] + psqt(pt, s, c);
        score += c == WHITE ? v : -v;
    }

    if (popcount(pos.pieces[WHITE][BISHOP]) >= 2) score += 35;
    if (popcount(pos.pieces[BLACK][BISHOP]) >= 2) score -= 35;

    score += mobility_bonus(pos, WHITE) - mobility_bonus(pos, BLACK);
    score += passed_pawn_bonus(pos, WHITE) - passed_pawn_bonus(pos, BLACK);

    return score;
}

Value evaluate_hce(const Position& pos) {
    const Value white = evaluate_hce_white(pos);
    return pos.side == WHITE ? white : -white;
}

Value evaluate(const Position& pos, const NNUE::Accumulator* accumulator) {
    auto& entry = EvalCache[std::size_t(pos.key) & (EVAL_CACHE_SIZE - 1)];
    if (entry.generation == EvalCacheGeneration && entry.key == pos.key && entry.side == pos.side && entry.score != VALUE_NONE) {
        ++EvalCacheHits;
        return entry.score;
    }

    const Value nn = NNUE::BerserkNNUE::instance().evaluate(pos, accumulator);
    const Value score = nn != VALUE_NONE ? nn : evaluate_hce(pos);
    entry = EvalCacheEntry{pos.key, pos.side, EvalCacheGeneration, score};
    return score;
}

Value evaluate_white(const Position& pos) {
    const Value side = evaluate(pos);
    return pos.side == WHITE ? side : -side;
}

bool init_backends() {
    bool ok = NNUE::BerserkNNUE::instance().try_load_default();
    Leela::RawBackend::instance().try_load_default();
    return ok;
}

bool load_berserk_nnue(const std::string& path) {
    clear_eval_cache();
    return NNUE::BerserkNNUE::instance().load(path);
}

void set_berserk_nnue_enabled(bool enabled) {
    clear_eval_cache();
    NNUE::BerserkNNUE::instance().set_enabled(enabled);
}

std::string backend_status() {
    const auto& b = NNUE::BerserkNNUE::instance().status();
    const auto& l = Leela::RawBackend::instance().status();
    std::string out = "BerserkNNUE=";
    out += (b.enabled ? "on" : "off");
    out += b.loaded ? "/loaded" : "/fallback";
    if (!b.path.empty()) out += " path=" + b.path;
    out += " evals=" + std::to_string(b.evals)
        + " rebuilds=" + std::to_string(b.accumulator_rebuilds)
        + " accUpdates=" + std::to_string(b.accumulator_updates)
        + " avg_ns=" + std::to_string(b.evals ? b.total_ns / b.evals : 0)
        + " cacheHits=" + std::to_string(EvalCacheHits);
    if (!b.message.empty()) out += " msg=" + b.message;
    out += " | LeelaRaw=";
    out += (l.enabled ? "on" : "off");
    out += l.loaded ? "/loaded" : "/fallback";
    if (!l.weights_path.empty()) out += " weights=" + l.weights_path;
    out += " channels=" + std::to_string(l.channels);
    out += " blocks=" + std::to_string(l.residual_blocks);
    out += " policy=" + std::to_string(l.policy_planes);
    out += " value=" + std::to_string(l.value_planes);
    out += " threads=" + std::to_string(l.threads) + (l.openmp ? "/omp" : "/single");
    out += " probes=" + std::to_string(l.probes) + " avg_us=" + std::to_string(l.probes ? l.total_us / l.probes : 0)
        + " batches=" + std::to_string(l.batch_calls) + "/" + std::to_string(l.batch_positions)
        + " batchWorkers=" + std::to_string(l.batch_workers)
        + " hits=" + std::to_string(l.cache_hits)
        + " cache=" + std::to_string(l.cache_entries) + "/" + std::to_string(l.cache_capacity)
        + " evict=" + std::to_string(l.cache_evictions)
        + " policyHits=" + std::to_string(l.policy_hits) + "/" + std::to_string(l.policy_queries)
        + " fail=" + std::to_string(l.failures);
    if (!l.message.empty()) out += " msg=" + l.message;
    const auto& s = Syzygy::Backend::instance().status();
    out += " | Syzygy=";
    out += s.enabled ? "on" : "off";
    out += s.loaded ? "/loaded" : "/unavailable";
    if (!s.path.empty()) out += " path=" + s.path;
    out += " largest=" + std::to_string(s.largest)
        + " limit=" + std::to_string(s.probe_limit)
        + " wdl=" + std::to_string(s.wdl_hits) + "/" + std::to_string(s.wdl_probes)
        + " root=" + std::to_string(s.root_hits) + "/" + std::to_string(s.root_probes)
        + " skip50=" + std::to_string(s.skipped_rule50)
        + " fail=" + std::to_string(s.failures);
    if (!s.message.empty()) out += " msg=" + s.message;
    return out;
}

} // namespace Scarlet::Eval

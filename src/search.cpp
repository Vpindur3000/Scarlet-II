#include "search.hpp"

#include "leela_raw.hpp"
#include "dag_backup.hpp"
#include "nnue_berserk.hpp"
#include "syzygy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Scarlet::Search {

namespace {

constexpr int INF = VALUE_INFINITE;
constexpr int NON_MATE_EVAL_CAP = 4000;
enum class TTDomain : std::uint8_t { ModernCorridor, HceSanitizer, ClassicAB };

Key tt_domain_key(Key key, TTDomain domain) {
    switch (domain) {
        case TTDomain::ModernCorridor: return key ^ 0xE7037ED1A0B428DBULL;
        case TTDomain::HceSanitizer:   return key ^ 0xA0761D6478BD642FULL;
        case TTDomain::ClassicAB:      return key;
    }
    return key;
}
// Syzygy proves WDL, not mate distance. Keep an exact, non-mate band so UCI
// reports a large cp outcome instead of inventing a forced mate length.
constexpr int SYZYGY_WIN_SCORE = 20000;

Options GlobalOptions{};

scarlet::tt::Bound to_tt_bound(Value best, Value originalAlpha, Value beta) {
    if (best >= beta) return scarlet::tt::Bound::Lower;
    if (best <= originalAlpha) return scarlet::tt::Bound::Upper;
    return scarlet::tt::Bound::Exact;
}

Value clamp_non_mate(int v) {
    return std::clamp(v, -NON_MATE_EVAL_CAP, NON_MATE_EVAL_CAP);
}

Value safe_eval(const Position& pos, const NNUE::Accumulator* accumulator = nullptr) {
    const Value v = Eval::evaluate(pos, accumulator);
    return is_mate_score(v) ? v : clamp_non_mate(v);
}

Value safe_hce_eval(const Position& pos) {
    const Value v = Eval::evaluate_hce(pos);
    return is_mate_score(v) ? v : clamp_non_mate(v);
}

void restrict_root_moves(MoveList& moves, const Limits& limits) {
    if (limits.searchmoves.empty()) return;
    int write = 0;
    for (int i = 0; i < moves.size; ++i) {
        const Move move = moves.moves[i];
        if (std::find(limits.searchmoves.begin(), limits.searchmoves.end(), move_to_tt(move))
                != limits.searchmoves.end())
            moves.moves[write++] = move;
    }
    moves.size = write;
}

int piece_value(PieceType pt) {
    static constexpr std::array<int, PIECE_TYPE_NB> values = {0, 100, 320, 330, 500, 900, 20000};
    return values[pt];
}

int material_phase(const Position& pos) {
    return std::clamp(
        3 * (popcount(pos.pieces[WHITE][KNIGHT]) + popcount(pos.pieces[BLACK][KNIGHT])
           + popcount(pos.pieces[WHITE][BISHOP]) + popcount(pos.pieces[BLACK][BISHOP]))
      + 5 * (popcount(pos.pieces[WHITE][ROOK]) + popcount(pos.pieces[BLACK][ROOK]))
      + 10 * (popcount(pos.pieces[WHITE][QUEEN]) + popcount(pos.pieces[BLACK][QUEEN])),
        0, 64);
}

int tapered_ld2_weight(const Position& pos, int middlegameWeight, int endgameWeight, int upperBound) {
    const int phase = material_phase(pos);
    const int middlegame = std::clamp(middlegameWeight, 0, upperBound);
    const int endgame = std::clamp(endgameWeight, 0, upperBound);
    return (phase * middlegame + (64 - phase) * endgame + 32) / 64;
}

int capture_order_key(const Position& pos, Move m) {
    if (is_promotion(m)) return 100000 + int(promotion_type(m)) * 1000;
    const Piece victim = is_en_passant(m) ? make_piece(~pos.side, PAWN) : pos.board[to_sq(m)];
    const Piece attacker = pos.board[from_sq(m)];
    return (victim == EMPTY ? 0 : 16 * piece_value(type_of(victim))) - piece_value(type_of(attacker));
}

void order_search_moves(const Position& pos, MoveList& moves, TTMove hashMove) {
    std::stable_sort(moves.moves.begin(), moves.moves.begin() + moves.size, [&](Move a, Move b) {
        const bool ah = hashMove != TT_MOVE_NONE && same_move_core(a, hashMove);
        const bool bh = hashMove != TT_MOVE_NONE && same_move_core(b, hashMove);
        if (ah != bh) return ah;
        const bool an = is_capture(a) || is_promotion(a);
        const bool bn = is_capture(b) || is_promotion(b);
        if (an != bn) return an;
        return capture_order_key(pos, a) > capture_order_key(pos, b);
    });
}

bool insufficient_material(const Position& pos) {
    if (pos.pieces[WHITE][PAWN] || pos.pieces[BLACK][PAWN]
        || pos.pieces[WHITE][ROOK] || pos.pieces[BLACK][ROOK]
        || pos.pieces[WHITE][QUEEN] || pos.pieces[BLACK][QUEEN])
        return false;

    const int knights = popcount(pos.pieces[WHITE][KNIGHT] | pos.pieces[BLACK][KNIGHT]);
    const Bitboard bishops = pos.pieces[WHITE][BISHOP] | pos.pieces[BLACK][BISHOP];
    const int bishopCount = popcount(bishops);
    if (knights + bishopCount <= 1) return true;
    if (knights != 0) return false;

    bool light = false, dark = false;
    Bitboard b = bishops;
    while (b) {
        const int sq = pop_lsb(b);
        ((file_of(sq) + rank_of(sq)) & 1 ? dark : light) = true;
    }
    return !(light && dark);
}

bool draw_by_rule(const Position& pos, bool atRoot = false) {
    const bool repetition = atRoot ? pos.repetition_count() >= 2 : pos.is_repetition();
    return pos.halfmoveClock >= 100 || repetition || insufficient_material(pos);
}

std::optional<Value> syzygy_score(const Position& pos) {
    const auto wdl = Syzygy::Backend::instance().probe_wdl(pos);
    if (!wdl) return std::nullopt;
    switch (*wdl) {
        case Syzygy::Wdl::Win:  return SYZYGY_WIN_SCORE;
        case Syzygy::Wdl::Loss: return -SYZYGY_WIN_SCORE;
        // A cursed outcome is not a safe exact win/loss at an arbitrary
        // 50-move clock. Treat it as draw here; DTZ resolves it at root.
        case Syzygy::Wdl::BlessedLoss:
        case Syzygy::Wdl::Draw:
        case Syzygy::Wdl::CursedWin: return VALUE_DRAW;
    }
    return std::nullopt;
}

Value syzygy_root_score(Syzygy::Wdl wdl) {
    switch (wdl) {
        case Syzygy::Wdl::Win:  return SYZYGY_WIN_SCORE;
        case Syzygy::Wdl::Loss: return -SYZYGY_WIN_SCORE;
        // DTZ reports these as 50-move-rule-contingent outcomes. Do not
        // advertise them as forced wins/losses to a GUI.
        case Syzygy::Wdl::BlessedLoss:
        case Syzygy::Wdl::Draw:
        case Syzygy::Wdl::CursedWin: return VALUE_DRAW;
    }
    return VALUE_DRAW;
}

int hce_danger(const Position& pos) {
    int danger = pos.in_check(pos.side) ? 72 : 0;
    MoveList noisy;
    generate_captures(pos, noisy);
    danger += std::min(56, noisy.size * 4);

    for (Color c : {WHITE, BLACK}) {
        for (PieceType pt : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN}) {
            Bitboard pieces = pos.pieces[c][pt];
            while (pieces) {
                const int sq = pop_lsb(pieces);
                const bool attacked = pos.square_attacked_by(sq, ~c);
                const bool defended = pos.square_attacked_by(sq, c);
                if (attacked && !defended) danger += std::min(44, piece_value(pt) / 20);
                else if (attacked && pt >= ROOK) danger += 7;
            }
        }

        const int king = pos.kingSq[c];
        const Bitboard zone = Attacks::King[king] | sq_bb(king);
        int pressure = 0;
        Bitboard pieces = pos.pieces[~c][KNIGHT];
        while (pieces) pressure += popcount(Attacks::Knight[pop_lsb(pieces)] & zone);
        pieces = pos.pieces[~c][BISHOP];
        while (pieces) pressure += popcount(Attacks::bishop_attacks(pop_lsb(pieces), pos.occAll) & zone);
        pieces = pos.pieces[~c][ROOK];
        while (pieces) pressure += popcount(Attacks::rook_attacks(pop_lsb(pieces), pos.occAll) & zone);
        pieces = pos.pieces[~c][QUEEN];
        while (pieces) pressure += popcount(Attacks::queen_attacks(pop_lsb(pieces), pos.occAll) & zone);
        danger += std::min(52, pressure * 7);
    }
    return std::clamp(danger, 0, 255);
}

bool tactically_noisy(Position& pos, Move m) {
    return is_capture(m) || is_promotion(m) || pos.in_check(pos.side) || gives_check(pos, m);
}

double heuristic_policy(Position& pos, Move m) {
    double score = 1.0;
    const Piece pc = pos.board[from_sq(m)];
    if (is_capture(m)) score += 4.0 + std::max(0, capture_order_key(pos, m)) / 160.0;
    if (is_promotion(m)) score += 10.0;
    if (is_castle(m)) score += 2.2;
    if (gives_check(pos, m)) score += 3.0;
    const int f = file_of(to_sq(m));
    const int r = rank_of(to_sq(m));
    const int centre = std::min(std::abs(f - 3), std::abs(f - 4))
                     + std::min(std::abs(r - 3), std::abs(r - 4));
    score += std::max(0, 3 - centre) * 0.35;
    if (pc != EMPTY && type_of(pc) == PAWN) score += 0.25;
    return std::max(0.05, score);
}

struct Corridor {
    Value lower = -NON_MATE_EVAL_CAP;
    Value upper = NON_MATE_EVAL_CAP;

    [[nodiscard]] Value center() const {
        return Value((static_cast<long long>(lower) + static_cast<long long>(upper)) / 2);
    }
    [[nodiscard]] int width() const { return std::max(0, int(upper) - int(lower)); }
    [[nodiscard]] bool valid() const { return lower <= upper; }
};

Corridor negate_corridor(const Corridor& c) {
    return Corridor{-c.upper, -c.lower};
}

enum class ProofObjective : std::uint8_t { TightenLower, TightenUpper };

// A Modern-B* node is shareable only when every search-relevant part of the
// state agrees. `Position::key` alone is insufficient: draw adjudication uses
// the reversible key history, and LD2 consumes eight historical frames plus
// the halfmove clock. The hash is an index; `same_proof_context()` below is
// the collision-safe equality check.
Key mix_proof_context(Key seed, Key value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return seed ^ (value ^ (value >> 31) ^ (seed << 7) ^ (seed >> 3));
}

std::size_t reversible_history_begin(const Position& pos) {
    if (pos.keyHistory.empty()) return 0;
    const std::size_t reversible = std::min<std::size_t>(
        std::max(0, pos.halfmoveClock), pos.keyHistory.size() - 1);
    return pos.keyHistory.size() - 1 - reversible;
}

Key proof_context_hash(const Position& pos) {
    Key h = mix_proof_context(0xD6E8FEB86659FD93ULL, pos.key);
    h = mix_proof_context(h, Key(std::max(0, pos.halfmoveClock)));
    const std::size_t begin = reversible_history_begin(pos);
    h = mix_proof_context(h, Key(pos.keyHistory.size() - begin));
    for (std::size_t i = begin; i < pos.keyHistory.size(); ++i)
        h = mix_proof_context(h, pos.keyHistory[i]);

    h = mix_proof_context(h, Key(pos.leelaHistoryCount));
    for (int i = 0; i < pos.leelaHistoryCount; ++i) {
        const LeelaHistoryFrame& frame = pos.leelaHistory[i];
        for (Bitboard pieces : frame.pieces)
            h = mix_proof_context(h, pieces);
        h = mix_proof_context(h, frame.repeated ? 1 : 0);
    }
    return h;
}

bool same_proof_context(const Position& a, const Position& b) {
    if (a.key != b.key || a.halfmoveClock != b.halfmoveClock
        || a.leelaHistoryCount != b.leelaHistoryCount)
        return false;

    const std::size_t aBegin = reversible_history_begin(a);
    const std::size_t bBegin = reversible_history_begin(b);
    if (a.keyHistory.size() - aBegin != b.keyHistory.size() - bBegin)
        return false;
    for (std::size_t ai = aBegin, bi = bBegin; ai < a.keyHistory.size(); ++ai, ++bi)
        if (a.keyHistory[ai] != b.keyHistory[bi]) return false;

    for (int i = 0; i < a.leelaHistoryCount; ++i) {
        if (a.leelaHistory[i].repeated != b.leelaHistory[i].repeated
            || a.leelaHistory[i].pieces != b.leelaHistory[i].pieces)
            return false;
    }
    return true;
}

struct ProofEdge {
    int child = -1;
    Move move = MOVE_NONE;
    double prior = 1.0;
    int visits = 0;
    bool noisy = false;
};

struct ProofCandidate {
    Move move = MOVE_NONE;
    double prior = 0.0;
    bool noisy = false;
};

enum class ExpansionState : std::uint8_t { Unexpanded, Partial, Full };

struct ProofNode {
    int positionId = -1;
    // Canonical path is used only to reconstruct a temporary NNUE accumulator
    // when this DAG node is expanded. Parent backup remains multi-parent.
    int accumulatorParent = -1;
    Move accumulatorIncoming = MOVE_NONE;
    // The ply is identical for context-equivalent transpositions generated
    // from one root. Parent links are retained for reverse DAG backup.
    int ply = 0;

    Corridor interval{};
    Corridor localInterval{};
    Value primary = VALUE_NONE;
    Value leela = VALUE_NONE;
    Value tactical = VALUE_NONE;
    Value fused = VALUE_ZERO;
    int danger = 0;
    int confidence = 0;
    int visits = 0;
    int subtreeDepth = 0;
    int bestEdge = -1;
    // The Modern-B* corridor is heuristic evidence, so a cached best move may
    // order this node but can never create a minimax cutoff.
    TTMove ttMove = TT_MOVE_NONE;

    bool evaluated = false;
    bool hasLeela = false;
    bool tacticalVerified = false;
    ExpansionState expansion = ExpansionState::Unexpanded;
    bool terminal = false;
    bool saturated = false;
    bool noisy = false;

    std::vector<Leela::PolicyEntry> policy;
    std::vector<ProofCandidate> candidates;
    std::vector<ProofEdge> children;
    std::vector<int> parents;
    std::size_t nextCandidate = 0;
    std::uint32_t localRevision = 0;
    std::uint32_t verifiedRevision = 0;
    int verifiedDepth = 0;
};

struct ModernStats {
    int frontierIterations = 0;
    int frontierExpansions = 0;
    int leelaRefinements = 0;
    int tacticalRefinements = 0;
    int abortedVerifications = 0;
    int leaderChanges = 0;
    int maxTreePly = 0;
    int maxVerificationPly = 0;
    std::uint64_t primaryEvals = 0;
    std::uint64_t transpositionHits = 0;
    std::uint64_t ttOrderHits = 0;
    std::uint64_t syzygyLeaves = 0;
};

double move_decision(const ProofNode& child, const ProofEdge& edge) {
    const Corridor rootView = negate_corridor(child.interval);
    const double fused = child.expansion != ExpansionState::Unexpanded
        ? double(rootView.center()) : -double(child.fused);
    return 0.43 * double(rootView.lower)
         + 0.34 * fused
         + 0.23 * double(rootView.upper)
         + 260.0 * edge.prior
         - 0.12 * double(rootView.width())
         - 0.16 * double(child.danger)
         - (child.hasLeela ? 0.0 : 32.0);
}

std::string bool01(bool x) { return x ? "1" : "0"; }

} // namespace

Options& options() { return GlobalOptions; }

void set_modern_bstar_enabled(bool enabled) { GlobalOptions.modern_bstar = enabled; }

Searcher::Searcher(std::size_t hash_mb) { set_hash(hash_mb); }

void Searcher::set_hash(std::size_t hash_mb) {
    scarlet::tt::Config cfg;
    cfg.hash_mb = std::max<std::size_t>(1, hash_mb);
    tt_.resize(cfg);
}

void Searcher::clear_hash() { tt_.clear(); }
void Searcher::stop() { control_.request_stop(); }
void Searcher::reset_stop() {
    control_.reset();
}
void Searcher::ponderhit() { control_.ponderhit_requested.store(true, std::memory_order_relaxed); }
void Searcher::set_info_callback(InfoCallback callback) {
    std::lock_guard<std::mutex> lock(info_mutex_);
    info_callback_ = std::move(callback);
}
void Searcher::clear_info_callback() {
    std::lock_guard<std::mutex> lock(info_mutex_);
    info_callback_ = {};
}
int Searcher::hashfull() const { return tt_.hashfull(0); }

void Searcher::emit_info(std::string_view line) {
    InfoCallback callback;
    {
        std::lock_guard<std::mutex> lock(info_mutex_);
        callback = info_callback_;
    }
    if (callback) callback(line);
}

bool Searcher::should_stop() {
    if (control_.stop_requested()) return true;
    control_.nodes.store(nodes_, std::memory_order_relaxed);
    if (limits_.nodes && control_.nodes.load(std::memory_order_relaxed) >= limits_.nodes) {
        control_.request_stop();
        return true;
    }
    if (control_.deadline_expired()) {
        control_.request_stop();
        return true;
    }
    if (limits_.ponder && control_.ponderhit_requested.load(std::memory_order_relaxed)) {
        if (!ponderhit_active_) {
            ponderhit_start_ = Clock::now();
            ponderhit_active_ = true;
            control_.set_deadline_after(limits_.ponderhit_movetime_ms);
        }
        if (limits_.ponderhit_movetime_ms > 0) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - ponderhit_start_).count();
            if (elapsed >= limits_.ponderhit_movetime_ms) {
                control_.request_stop();
                return true;
            }
        }
    }
    return false;
}

Value Searcher::qsearch(Position& pos, Value alpha, Value beta, int ply,
                        bool& completed, bool hce_only, NNUE::Accumulator* accumulator) {
    if (!completed) return VALUE_ZERO;
    ++nodes_;
    if (should_stop()) {
        completed = false;
        return VALUE_ZERO;
    }
    if (ply >= MAX_PLY - 1) return hce_only ? safe_hce_eval(pos) : safe_eval(pos, accumulator);
    if (draw_by_rule(pos, ply == 0)) return VALUE_DRAW;
    if (ply > 0) {
        if (const auto tb = syzygy_score(pos)) return *tb;
    }

    const bool inCheck = pos.in_check(pos.side);
    MoveList moves;
    if (inCheck) {
        generate_evasions(pos, moves);
        if (moves.size == 0) return mated_in(ply);
    } else {
        const Value stand = hce_only ? safe_hce_eval(pos) : safe_eval(pos, accumulator);
        if (stand >= beta) return stand;
        if (stand > alpha) alpha = stand;
        generate_captures(pos, moves);
        if (moves.size == 0) {
            MoveList legal;
            generate_legal(pos, legal);
            return legal.size == 0 ? VALUE_DRAW : alpha;
        }
    }

    order_search_moves(pos, moves, TT_MOVE_NONE);
    int legalSearched = 0;
    for (Move m : moves) {
        StateInfo st;
        if (!pos.make_move(m, st)) continue;
        ++legalSearched;
        NNUE::Accumulator childAccumulator;
        NNUE::Accumulator* childPtr = nullptr;
        if (!hce_only && accumulator) {
            childAccumulator = *accumulator;
            NNUE::BerserkNNUE::instance().apply_move(childAccumulator, pos, m, st);
            childPtr = &childAccumulator;
        }
        const Value score = -qsearch(pos, -beta, -alpha, ply + 1, completed, hce_only, childPtr);
        pos.undo_move(m, st);
        if (!completed) return VALUE_ZERO;
        if (score >= beta) return score;
        if (score > alpha) alpha = score;
    }
    if (!inCheck && legalSearched == 0) {
        MoveList legal;
        generate_legal(pos, legal);
        if (legal.size == 0) return VALUE_DRAW;
    }
    return alpha;
}

Value Searcher::negamax(Position& pos, int depth, Value alpha, Value beta, int ply,
                        bool& completed, bool hce_only, NNUE::Accumulator* accumulator) {
    if (!completed) return VALUE_ZERO;
    if (depth <= 0) return qsearch(pos, alpha, beta, ply, completed, hce_only, accumulator);
    if (ply >= MAX_PLY - 1) return hce_only ? safe_hce_eval(pos) : safe_eval(pos, accumulator);
    if (draw_by_rule(pos, ply == 0)) return VALUE_DRAW;
    if (ply > 0) {
        if (const auto tb = syzygy_score(pos)) return *tb;
    }

    ++nodes_;
    if (should_stop()) {
        completed = false;
        return VALUE_ZERO;
    }

    const Value originalAlpha = alpha;
    const Key ttKey = tt_domain_key(pos.key,
        hce_only ? TTDomain::HceSanitizer : TTDomain::ClassicAB);
    TTMove ttMove = TT_MOVE_NONE;
    if (auto probe = tt_.probe(ttKey, ply); probe.hit) {
        ttMove = probe.data.has_move ? probe.data.move : TT_MOVE_NONE;
        scarlet::tt::Value cutoff = scarlet::tt::VALUE_NONE;
        if (scarlet::tt::TranspositionTable::can_cutoff_classic(
                probe.data, depth, alpha, beta, cutoff))
            return cutoff;
    }

    MoveList moves;
    generate_legal(pos, moves);
    if (ply == 0) restrict_root_moves(moves, limits_);
    if (moves.size == 0) return pos.in_check(pos.side) ? mated_in(ply) : VALUE_DRAW;
    order_search_moves(pos, moves, ttMove);

    Value best = -VALUE_INFINITE;
    Move bestMove = MOVE_NONE;
    for (Move m : moves) {
        StateInfo st;
        if (!pos.make_move(m, st)) continue;
        NNUE::Accumulator childAccumulator;
        NNUE::Accumulator* childPtr = nullptr;
        if (!hce_only && accumulator) {
            childAccumulator = *accumulator;
            NNUE::BerserkNNUE::instance().apply_move(childAccumulator, pos, m, st);
            childPtr = &childAccumulator;
        }
        const Value score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1,
                                     completed, hce_only, childPtr);
        pos.undo_move(m, st);
        if (!completed) return VALUE_ZERO;

        if (score > best) {
            best = score;
            bestMove = m;
            if (ply == 0) root_best_ = m;
        }
        alpha = std::max(alpha, score);
        if (alpha >= beta) break;
    }

    if (best == -VALUE_INFINITE) best = VALUE_DRAW;
    tt_.save_classic(ttKey, best, to_tt_bound(best, originalAlpha, beta), depth,
                     move_to_tt(bestMove), Eval::evaluate_hce(pos), ply, false);
    return best;
}

Result Searcher::think_modern_bstar(Position& root, const Limits& limits) {
    limits_ = limits;
    start_ = Clock::now();
    if (!limits.ponder) control_.set_deadline_after(limits.movetime_ms);
    ponderhit_active_ = false;
    nodes_ = 0;
    root_best_ = MOVE_NONE;
    tt_.new_search();

    Result result;
    result.requested_depth = std::max(1, limits.depth);
    const auto& nnStatus = NNUE::BerserkNNUE::instance().status();
    const auto& ldStatus = Leela::RawBackend::instance().status();
    result.degraded = (nnStatus.enabled && !nnStatus.loaded)
                   || (GlobalOptions.use_ld2 && ldStatus.enabled && !ldStatus.loaded);
    ModernStats stats;
    const auto nnStart = NNUE::BerserkNNUE::instance().status();
    const auto ldStart = Leela::RawBackend::instance().status();

    MoveList rootLegal;
    generate_legal(root, rootLegal);
    restrict_root_moves(rootLegal, limits);
    if (rootLegal.size == 0) {
        result.score = root.in_check(root.side) ? mated_in(0) : VALUE_DRAW;
        result.lower = result.upper = result.score;
        result.confidence = 100;
        result.score_source = "terminal";
        return result;
    }
    if (draw_by_rule(root, true)) {
        result.best_move = rootLegal.moves[0];
        result.score = VALUE_DRAW;
        result.depth = result.seldepth = 1;
        result.nodes = 1;
        result.pv = move_to_uci(result.best_move);
        result.search_info = "draw by rule at root";
        result.lower = result.upper = result.score;
        result.confidence = 100;
        result.score_source = "rule-draw";
        return result;
    }
    if (rootLegal.size == 1) {
        result.best_move = rootLegal.moves[0];
        bool completed = true;
        NNUE::Accumulator accumulator;
        NNUE::BerserkNNUE::instance().refresh(accumulator, root);
        const int verifyDepth = std::max(2, GlobalOptions.tactical_depth);
        result.score = negamax(root, verifyDepth, -VALUE_INFINITE, VALUE_INFINITE,
                               0, completed, false, &accumulator);
        if (!completed) {
            Position child = root;
            StateInfo state;
            if (child.make_move(result.best_move, state)) result.score = -safe_eval(child);
        }
        result.depth = result.seldepth = completed ? verifyDepth : 1;
        result.nodes = std::max<std::uint64_t>(1, nodes_);
        result.pv = move_to_uci(result.best_move);
        result.lower = result.upper = result.score;
        result.confidence = completed ? 90 : 45;
        result.score_source = completed ? "forced-tactical" : "forced-primary";
        return result;
    }

    const int maxTreeDepth = std::clamp(std::max(2, limits.depth), 2, 64);
    const int nodeLimit = std::max(1024, GlobalOptions.proof_node_limit);
    int currentEpoch = 0;
    int activeNodeBudget = std::min(nodeLimit, 1024);
    int wideningTarget = 4;
    int currentVerificationDepth = std::max(1, GlobalOptions.tactical_depth - 1);
    const bool leelaAvailable = GlobalOptions.use_ld2
        && Leela::RawBackend::instance().status().enabled
        && Leela::RawBackend::instance().status().loaded;

    int leelaBudget = 0;
    if (leelaAvailable) {
        const int configured = 1 + std::max(0, GlobalOptions.ld2_max_root_probes)
                               + std::max(0, GlobalOptions.ld2_max_frontier_probes);
        if (limits.infinite) leelaBudget = configured * 32;
        else if (limits.movetime_ms > 0)
            leelaBudget = std::min(configured, std::max(2, limits.movetime_ms / 28 + 2));
        else leelaBudget = configured;
    }
    const int leelaBudgetStart = leelaBudget;

    std::vector<ProofNode> tree;
    // Expansion keeps references to the current node while appending all of
    // its children, so reserve the complete proof-tree budget up front.
    tree.reserve(nodeLimit);
    tree.push_back(ProofNode{});
    std::vector<Position> positionArena;
    positionArena.reserve(nodeLimit);
    positionArena.push_back(root);
    tree[0].positionId = 0;
    tree[0].ply = 0;

    // Hash buckets retain all candidates and validate the complete context
    // before sharing a node, so a 64-bit hash collision remains harmless.
    std::unordered_map<Key, std::vector<int>> transpositions;
    transpositions.reserve(std::size_t(nodeLimit) * 2);
    transpositions[proof_context_hash(root)].push_back(0);

    auto intern_node = [&](Position&& pos, int ply) -> std::pair<int, bool> {
        const Key context = proof_context_hash(pos);
        auto& bucket = transpositions[context];
        for (const int candidate : bucket) {
            // Mate distance and the configured proof depth are root-relative,
            // so never merge otherwise identical contexts reached at a
            // different ply after an earlier irreversible sequence.
            if (tree[candidate].ply == ply
                    && same_proof_context(positionArena[tree[candidate].positionId], pos)) {
                ++stats.transpositionHits;
                return {candidate, false};
            }
        }
        if (int(tree.size()) >= activeNodeBudget) return {-1, false};
        ProofNode node;
        positionArena.push_back(std::move(pos));
        node.positionId = int(positionArena.size()) - 1;
        node.ply = ply;
        tree.push_back(std::move(node));
        const int idx = int(tree.size()) - 1;
        bucket.push_back(idx);
        return {idx, true};
    };

    struct AccumulatorCacheEntry {
        NNUE::Accumulator accumulator;
        std::list<int>::iterator lru;
    };
    constexpr std::size_t accumulatorCacheCapacity = 2048;
    std::list<int> accumulatorLru;
    std::unordered_map<int, AccumulatorCacheEntry> accumulatorCache;
    accumulatorCache.reserve(accumulatorCacheCapacity);

    auto cache_accumulator = [&](int idx, const NNUE::Accumulator& accumulator) {
        if (!accumulator.valid) return;
        if (auto it = accumulatorCache.find(idx); it != accumulatorCache.end()) {
            it->second.accumulator = accumulator;
            accumulatorLru.splice(accumulatorLru.end(), accumulatorLru, it->second.lru);
            it->second.lru = std::prev(accumulatorLru.end());
            return;
        }
        if (accumulatorCache.size() >= accumulatorCacheCapacity) {
            const int victim = accumulatorLru.front();
            accumulatorLru.pop_front();
            accumulatorCache.erase(victim);
        }
        accumulatorLru.push_back(idx);
        accumulatorCache.emplace(idx, AccumulatorCacheEntry{accumulator, std::prev(accumulatorLru.end())});
    };

    auto build_accumulator_for_node = [&](int idx, NNUE::Accumulator& accumulator) {
        std::vector<int> path;
        int checkpoint = idx;
        for (int current = idx; current > 0; current = tree[current].accumulatorParent) {
            if (auto cached = accumulatorCache.find(current); cached != accumulatorCache.end()) {
                accumulator = cached->second.accumulator;
                accumulatorLru.splice(accumulatorLru.end(), accumulatorLru, cached->second.lru);
                cached->second.lru = std::prev(accumulatorLru.end());
                checkpoint = current;
                break;
            }
            if (tree[current].accumulatorParent < 0) {
                // A malformed canonical link must not corrupt evaluation.
                NNUE::BerserkNNUE::instance().refresh(
                    accumulator, positionArena[tree[idx].positionId]);
                cache_accumulator(idx, accumulator);
                return;
            }
            path.push_back(current);
            checkpoint = 0;
        }

        if (checkpoint == 0) {
            if (auto cachedRoot = accumulatorCache.find(0); cachedRoot != accumulatorCache.end())
                accumulator = cachedRoot->second.accumulator;
            else
                NNUE::BerserkNNUE::instance().refresh(accumulator, positionArena[0]);
        }
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            const ProofNode& child = tree[*it];
            if (*it == checkpoint) continue;
            Position parent = positionArena[tree[child.accumulatorParent].positionId];
            StateInfo state;
            if (!parent.make_move(child.accumulatorIncoming, state)) {
                NNUE::BerserkNNUE::instance().refresh(
                    accumulator, positionArena[tree[idx].positionId]);
                cache_accumulator(idx, accumulator);
                return;
            }
            NNUE::BerserkNNUE::instance().apply_move(accumulator, parent,
                                                      child.accumulatorIncoming, state);
        }
        cache_accumulator(idx, accumulator);
    };

    NNUE::Accumulator rootAccumulatorCache;
    NNUE::BerserkNNUE::instance().refresh(rootAccumulatorCache, positionArena[0]);
    cache_accumulator(0, rootAccumulatorCache);

    auto elapsed_ms = [&]() -> std::uint64_t {
        return std::uint64_t(std::max<long long>(0,
            std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_).count()));
    };

    auto estimated_leela_ms = [&]() {
        const auto& status = Leela::RawBackend::instance().status();
        if (status.probes > 0)
            return std::clamp<int>(int(status.total_us / status.probes / 1000) + 12, 20, 5000);
        // A cold LD2 CPU pass is orders of magnitude dearer than an NNUE leaf.
        // Start conservatively, then replace this with the measured runtime.
        // The supported AVX2/OpenMP build normally lands well below this;
        // leave headroom for cold caches and slower four-core systems.
        return 35;
    };

    auto enough_time_for_leela = [&]() {
        if (limits.infinite || limits.movetime_ms <= 0) return true;
        return int(elapsed_ms()) + estimated_leela_ms() + 20 < limits.movetime_ms;
    };

    auto install_leela_probe = [](ProofNode& node, const Leela::RawProbeResult& probe) {
        node.leela = probe.cp;
        node.policy = probe.policy;
        node.hasLeela = true;
    };

    auto probe_leela = [&](int idx) -> bool {
        ProofNode& node = tree[idx];
        if (node.hasLeela || !leelaAvailable) return node.hasLeela;

        const Position& position = positionArena[node.positionId];
        auto cached = Leela::RawBackend::instance().cached_probe(position);
        std::optional<Leela::RawProbeResult> probe;
        if (cached && cached->valid) {
            probe = std::move(cached);
        } else {
            if (leelaBudget <= 0 || should_stop() || !enough_time_for_leela()) return false;
            const auto before = Leela::RawBackend::instance().status().probes;
            probe = Leela::RawBackend::instance().probe(position,
                node.primary == VALUE_NONE ? VALUE_ZERO : node.primary, control_.token());
            if (Leela::RawBackend::instance().status().probes > before) --leelaBudget;
        }
        if (!probe || !probe->valid) return false;
        install_leela_probe(node, *probe);
        return true;
    };

    // Root siblings are independent raw-network jobs. Submit them together so
    // the persistent LD2 pool can use available cores across positions. This
    // is deliberately separate from the scalar frontier probe: it avoids
    // queueing a one-element "batch" on every proof iteration.
    auto probe_leela_batch = [&](const std::vector<int>& indices) {
        std::vector<int> resolved;
        if (!leelaAvailable || indices.empty()) return resolved;

        std::vector<Leela::RawProbeRequest> requests;
        std::vector<int> requestNodes;
        requests.reserve(indices.size());
        requestNodes.reserve(indices.size());
        for (const int idx : indices) {
            if (idx < 0 || idx >= int(tree.size())) continue;
            ProofNode& node = tree[idx];
            if (node.hasLeela) {
                resolved.push_back(idx);
                continue;
            }
            const Position& position = positionArena[node.positionId];
            if (auto cached = Leela::RawBackend::instance().cached_probe(position);
                    cached && cached->valid) {
                install_leela_probe(node, *cached);
                resolved.push_back(idx);
                continue;
            }
            requests.push_back(Leela::RawProbeRequest{
                position, node.primary == VALUE_NONE ? VALUE_ZERO : node.primary});
            requestNodes.push_back(idx);
        }
        if (requests.empty() || leelaBudget <= 0 || should_stop() || !enough_time_for_leela())
            return resolved;

        // Do not submit more uncached jobs than the proof driver's remaining
        // allowance. Cache hits above are free and remain usable.
        if (int(requests.size()) > leelaBudget) {
            requests.resize(std::size_t(leelaBudget));
            requestNodes.resize(std::size_t(leelaBudget));
        }
        const auto before = Leela::RawBackend::instance().status().probes;
        auto probes = Leela::RawBackend::instance().probe_batch(requests, control_.token());
        const auto after = Leela::RawBackend::instance().status().probes;
        leelaBudget = std::max(0, leelaBudget - int(after - before));
        for (std::size_t i = 0; i < probes.size(); ++i) {
            if (!probes[i] || !probes[i]->valid) continue;
            install_leela_probe(tree[requestNodes[i]], *probes[i]);
            resolved.push_back(requestNodes[i]);
        }
        return resolved;
    };

    std::function<void(int)> rebuild_leaf = [&](int idx) {
        ProofNode& node = tree[idx];
        if (node.terminal || node.expansion != ExpansionState::Unexpanded) return;
        const Corridor previousLocal = node.localInterval;
        if (node.primary == VALUE_NONE) {
            node.primary = safe_eval(positionArena[node.positionId]);
            ++stats.primaryEvals;
            ++nodes_;
        }
        if (node.tactical != VALUE_NONE && is_mate_score(node.tactical)) {
            node.interval = Corridor{node.tactical, node.tactical};
            node.localInterval = node.interval;
            node.fused = node.tactical;
            node.confidence = 100;
            node.evaluated = true;
            if (node.localInterval.lower != previousLocal.lower
                    || node.localInterval.upper != previousLocal.upper)
                ++node.localRevision;
            return;
        }

        const int leelaWeight = node.hasLeela
            ? tapered_ld2_weight(positionArena[node.positionId], GlobalOptions.leela_value_weight,
                                 GlobalOptions.leela_endgame_value_weight, 80)
            : 0;
        node.fused = node.hasLeela
            ? clamp_non_mate(((100 - leelaWeight) * int(node.primary) + leelaWeight * int(node.leela)) / 100)
            : node.primary;

        int lo = node.primary;
        int hi = node.primary;
        if (node.hasLeela) {
            lo = std::min(lo, int(node.leela));
            hi = std::max(hi, int(node.leela));
        }
        if (node.tacticalVerified) {
            lo = std::min(lo, int(node.tactical));
            hi = std::max(hi, int(node.tactical));
        }

        int margin = node.hasLeela ? 22 : 112;
        margin += node.danger / (node.hasLeela ? 3 : 2);
        if (node.hasLeela) margin += std::abs(int(node.primary) - int(node.leela)) / 7;
        if (node.tacticalVerified) {
            // A completed full-window 3-4 ply sanitizer is much stronger
            // evidence than a raw fallback leaf. Keep disagreement in the
            // min/max span, but shrink the extra uncertainty halo.
            margin = 18 + node.danger / 5;
            if (node.hasLeela)
                margin += std::abs(int(node.tactical) - int(node.leela)) / 12;
        }
        node.interval = Corridor{clamp_non_mate(lo - margin), clamp_non_mate(hi + margin)};
        node.localInterval = node.interval;
        node.confidence = std::clamp(30 + (node.hasLeela ? 28 : 0)
                                     + (node.tacticalVerified ? 22 : 0) - node.danger / 10, 8, 95);
        node.evaluated = true;
        if (node.localInterval.lower != previousLocal.lower
                || node.localInterval.upper != previousLocal.upper)
            ++node.localRevision;
        tt_.save_corridor(tt_domain_key(proof_context_hash(positionArena[node.positionId]), TTDomain::ModernCorridor), node.interval.lower, node.interval.upper, node.subtreeDepth,
                          TT_MOVE_NONE, node.ply, false, node.primary);
    };

    auto evaluate_leaf = [&](int idx, bool wantLeela, const NNUE::Accumulator* accumulator = nullptr) {
        ProofNode& node = tree[idx];
        const Position& position = positionArena[node.positionId];
        if (draw_by_rule(position, idx == 0)) {
            node.interval = Corridor{VALUE_DRAW, VALUE_DRAW};
            node.localInterval = node.interval;
            node.primary = node.leela = node.fused = VALUE_DRAW;
            node.evaluated = node.terminal = node.saturated = true;
            node.confidence = 100;
            return;
        }
        // Root DTZ is handled before either search core starts so it can
        // return an actual optimal move. Interior WDL values, however, are
        // exact terminal bounds and need no NNUE/LD2 corridor.
        if (idx != 0) {
            if (const auto tb = syzygy_score(position)) {
                node.interval = Corridor{*tb, *tb};
                node.localInterval = node.interval;
                node.primary = node.fused = *tb;
                node.evaluated = node.terminal = node.saturated = true;
                node.confidence = 100;
                ++stats.syzygyLeaves;
                ++nodes_;
                return;
            }
        }
        if (auto cached = tt_.probe(tt_domain_key(proof_context_hash(position), TTDomain::ModernCorridor), node.ply);
                cached.hit && cached.data.has_move) {
            node.ttMove = cached.data.move;
            ++stats.ttOrderHits;
        }
        NNUE::Accumulator rebuilt;
        if (!accumulator) {
            build_accumulator_for_node(idx, rebuilt);
            accumulator = &rebuilt;
        }
        node.primary = safe_eval(position, accumulator);
        node.danger = hce_danger(position);
        ++stats.primaryEvals;
        ++nodes_;
        if (wantLeela) probe_leela(idx);
        rebuild_leaf(idx);
    };

    auto policy_prior = [](const ProofNode& node, Move move) {
        const TTMove core = move_to_tt(move);
        for (const auto& p : node.policy)
            if (same_move_core(p.move, core)) return double(p.prior);
        return 0.0;
    };

    std::function<bool(int)> backup_node = [&](int idx) {
        ProofNode& node = tree[idx];
        if (node.expansion == ExpansionState::Unexpanded || node.children.empty()) return false;
        const Corridor oldInterval = node.interval;
        const int oldBestEdge = node.bestEdge;
        const int oldDepth = node.subtreeDepth;
        const bool oldSaturated = node.saturated;

        int lower = -VALUE_INFINITE;
        int upper = -VALUE_INFINITE;
        int bestEdge = -1;
        double bestDecision = -1e100;
        int subtreeDepth = 0;
        bool saturated = node.expansion == ExpansionState::Full;
        for (int edgeIdx = 0; edgeIdx < int(node.children.size()); ++edgeIdx) {
            const ProofEdge& edge = node.children[edgeIdx];
            const ProofNode& child = tree[edge.child];
            const Corridor move = negate_corridor(child.interval);
            lower = std::max(lower, int(move.lower));
            upper = std::max(upper, int(move.upper));
            const double decision = 0.48 * double(move.lower) + 0.34 * double(move.center())
                                  + 0.18 * double(move.upper) + 220.0 * edge.prior
                                  - 0.10 * double(move.width());
            if (decision > bestDecision) {
                bestDecision = decision;
                bestEdge = edgeIdx;
            }
            subtreeDepth = std::max(subtreeDepth, 1 + child.subtreeDepth);
            saturated = saturated && (child.saturated || child.terminal);
        }
        const Corridor backed{Value(lower), Value(std::max(lower, upper))};
        if (node.evaluated && node.localInterval.valid()) {
            const int overlapLower = std::max(int(backed.lower), int(node.localInterval.lower));
            const int overlapUpper = std::min(int(backed.upper), int(node.localInterval.upper));
            if (overlapLower <= overlapUpper) {
                // The local dual-evaluator corridor is the prior hypothesis;
                // child backup is new proof evidence. Their overlap is the
                // refined working corridor.
                node.interval = Corridor{Value(overlapLower), Value(overlapUpper)};
            } else {
                // Contradictory evidence is uncertainty, not a licence to
                // silently pick one source. Preserve the complete hull.
                node.interval = Corridor{
                    Value(std::min(int(backed.lower), int(node.localInterval.lower))),
                    Value(std::max(int(backed.upper), int(node.localInterval.upper)))};
            }
        } else {
            node.interval = backed;
        }
        if (node.expansion != ExpansionState::Full && node.localInterval.valid()) {
            // Deferred legal moves are still unresolved evidence. Until they
            // are materialized, keep the parent's local corridor in the hull.
            node.interval = Corridor{
                Value(std::min(int(node.interval.lower), int(node.localInterval.lower))),
                Value(std::max(int(node.interval.upper), int(node.localInterval.upper)))};
        }
        node.bestEdge = bestEdge;
        node.subtreeDepth = subtreeDepth;
        node.saturated = saturated;
        const TTMove bestMove = bestEdge >= 0 ? move_to_tt(node.children[bestEdge].move) : TT_MOVE_NONE;
        tt_.save_corridor(tt_domain_key(proof_context_hash(positionArena[node.positionId]), TTDomain::ModernCorridor), node.interval.lower, node.interval.upper, node.subtreeDepth,
                          bestMove, node.ply, false,
                          node.primary == VALUE_NONE ? node.interval.center() : node.primary);
        return node.interval.lower != oldInterval.lower || node.interval.upper != oldInterval.upper
            || node.bestEdge != oldBestEdge || node.subtreeDepth != oldDepth
            || node.saturated != oldSaturated;
    };

    auto backup_ancestors = [&](int idx) {
        propagate_dag_change(tree.size(), idx,
            [&](int node) { return tree[node].ply; },
            [&](int node) -> const std::vector<int>& { return tree[node].parents; },
            backup_node);
    };

    std::function<bool(int, bool)> expand_node = [&](int idx, bool rootExpansion) -> bool {
        if (idx < 0 || idx >= int(tree.size())) return false;
        ProofNode& node = tree[idx];
        if (node.expansion == ExpansionState::Full || node.terminal || node.saturated) return false;
        if (node.ply >= maxTreeDepth) {
            node.saturated = true;
            backup_ancestors(idx);
            return false;
        }

        NNUE::Accumulator nodeAccumulator;
        build_accumulator_for_node(idx, nodeAccumulator);
        if (!node.evaluated) evaluate_leaf(idx, rootExpansion, &nodeAccumulator);
        if (node.terminal || node.saturated) {
            backup_ancestors(idx);
            return true;
        }
        if (leelaAvailable && !node.hasLeela
                && (!rootExpansion || GlobalOptions.ld2_root_policy_always)) {
            probe_leela(idx);
            rebuild_leaf(idx);
        }

        if (node.candidates.empty()) {
            MoveList legal;
            Position& position = positionArena[node.positionId];
            generate_legal(position, legal);
            if (rootExpansion) restrict_root_moves(legal, limits);
            if (legal.size == 0) {
                node.interval = position.in_check(position.side)
                    ? Corridor{mated_in(node.ply), mated_in(node.ply)}
                    : Corridor{VALUE_DRAW, VALUE_DRAW};
                node.localInterval = node.interval;
                node.fused = node.interval.center();
                node.terminal = node.saturated = node.evaluated = true;
                node.expansion = ExpansionState::Full;
                node.confidence = 100;
                backup_ancestors(idx);
                return true;
            }

            std::vector<double> heuristic(legal.size, 0.0);
            double heuristicSum = 0.0;
            for (int i = 0; i < legal.size; ++i) {
                heuristic[i] = heuristic_policy(position, legal.moves[i]);
                heuristicSum += heuristic[i];
            }
            heuristicSum = std::max(heuristicSum, 1e-9);
            const double policyWeight = node.hasLeela
                ? tapered_ld2_weight(position, GlobalOptions.leela_policy_weight,
                                     GlobalOptions.leela_endgame_policy_weight, 90) / 100.0
                : 0.0;
            node.candidates.reserve(legal.size);
            for (int i = 0; i < legal.size; ++i) {
                const double h = heuristic[i] / heuristicSum;
                const double l = policy_prior(node, legal.moves[i]);
                const double prior = l > 0.0 ? policyWeight * l + (1.0 - policyWeight) * h : h;
                node.candidates.push_back(ProofCandidate{
                    legal.moves[i], std::max(0.0001, prior),
                    tactically_noisy(position, legal.moves[i])});
            }
            double priorSum = 0.0;
            for (const auto& c : node.candidates) priorSum += c.prior;
            for (auto& c : node.candidates) c.prior /= std::max(priorSum, 1e-12);
            const TTMove ttMove = node.ttMove;
            std::stable_sort(node.candidates.begin(), node.candidates.end(),
                [ttMove](const ProofCandidate& a, const ProofCandidate& b) {
                    const bool aHash = ttMove != TT_MOVE_NONE && same_move_core(a.move, ttMove);
                    const bool bHash = ttMove != TT_MOVE_NONE && same_move_core(b.move, ttMove);
                    if (aHash != bHash) return aHash;
                    return a.prior > b.prior;
                });
            node.children.reserve(node.candidates.size());
        }

        const std::size_t materializeLimit = rootExpansion
            ? node.candidates.size()
            : std::min(node.candidates.size(), node.nextCandidate + std::size_t(wideningTarget));
        bool materialized = false;
        while (node.nextCandidate < materializeLimit) {
            const ProofCandidate c = node.candidates[node.nextCandidate];
            Position childPos = positionArena[node.positionId];
            StateInfo st;
            if (!childPos.make_move(c.move, st)) {
                ++node.nextCandidate;
                continue;
            }
            const auto [childIdx, inserted] = intern_node(std::move(childPos), node.ply + 1);
            if (childIdx < 0) break;
            ++node.nextCandidate;
            materialized = true;
            ProofNode& child = tree[childIdx];
            if (std::find(child.parents.begin(), child.parents.end(), idx) == child.parents.end())
                child.parents.push_back(idx);
            // Noisy is path-derived. Keeping the conservative OR at the
            // shared node can only request extra sanitation, never skip one.
            child.noisy = child.noisy || c.noisy;
            node.children.push_back(ProofEdge{childIdx, c.move, c.prior, 0, c.noisy});
            if (inserted) {
                child.accumulatorParent = idx;
                child.accumulatorIncoming = c.move;
                NNUE::Accumulator childAccumulator = nodeAccumulator;
                NNUE::BerserkNNUE::instance().apply_move(
                    childAccumulator, positionArena[child.positionId], c.move, st);
                cache_accumulator(childIdx, childAccumulator);
                evaluate_leaf(childIdx, false, &childAccumulator);
            }
            stats.maxTreePly = std::max(stats.maxTreePly, node.ply + 1);
        }

        // At root, LD2 directly evaluates the highest-policy candidate moves as
        // well as supplying root policy. This is intentionally substantial: the
        // beta is meant to demonstrate a genuine dual-evaluator root race.
        if (rootExpansion && leelaAvailable) {
            std::vector<int> rootProbeCandidates;
            rootProbeCandidates.reserve(node.children.size());
            int scheduled = 0;
            for (const ProofEdge& edge : node.children) {
                ProofNode& child = tree[edge.child];
                // A Syzygy-resolved child already has an exact WDL interval;
                // spending LD2 batch capacity on it cannot improve the node.
                if (child.terminal || child.saturated) continue;
                const bool highPolicy = edge.prior * 1000.0 >= GlobalOptions.ld2_min_policy_permille;
                if (scheduled >= GlobalOptions.ld2_max_root_probes && !highPolicy && !child.noisy) continue;
                // Tactically noisy root continuations enter the short AB
                // sanitizer first; do not spend an LD2 pass on hanging pieces.
                if (child.noisy || child.danger >= 72) continue;
                if (should_stop()) break;
                rootProbeCandidates.push_back(edge.child);
                ++scheduled;
            }
            const std::size_t batchSize = std::size_t(std::clamp(GlobalOptions.ld2_batch_size, 1, 32));
            for (std::size_t first = 0; first < rootProbeCandidates.size(); first += batchSize) {
                if (should_stop() || !enough_time_for_leela()) break;
                const std::size_t last = std::min(rootProbeCandidates.size(), first + batchSize);
                std::vector<int> batch(rootProbeCandidates.begin() + std::ptrdiff_t(first),
                                       rootProbeCandidates.begin() + std::ptrdiff_t(last));
                for (const int childIdx : probe_leela_batch(batch))
                    rebuild_leaf(childIdx);
                if (leelaBudget <= 0) break;
            }
        }

        if (node.children.empty()) node.expansion = ExpansionState::Unexpanded;
        else if (node.nextCandidate >= node.candidates.size()) node.expansion = ExpansionState::Full;
        else node.expansion = ExpansionState::Partial;
        if (node.expansion != ExpansionState::Full) node.saturated = false;
        if (materialized) ++stats.frontierExpansions;
        backup_ancestors(idx);
        return materialized;
    };

    evaluate_leaf(0, GlobalOptions.ld2_root_policy_always);
    // The root is a frontier too: establish a tactical baseline before legal
    // descriptors are materialized. This makes verify-before-expand an actual
    // invariant rather than an ordering that only applied below the root.
    bool rootVerificationCompleted = true;
    const int rootVerificationDepth = std::max(1, GlobalOptions.tactical_depth - 1);
    const Value rootTactical = negamax(positionArena[0], rootVerificationDepth,
                                       -VALUE_INFINITE, VALUE_INFINITE, 0,
                                       rootVerificationCompleted, true, nullptr);
    if (rootVerificationCompleted) {
        tree[0].tactical = rootTactical;
        tree[0].tacticalVerified = true;
        tree[0].verifiedDepth = rootVerificationDepth;
        rebuild_leaf(0);
        tree[0].verifiedRevision = tree[0].localRevision;
        ++stats.tacticalRefinements;
        stats.maxVerificationPly = std::max(stats.maxVerificationPly, rootVerificationDepth);
    }
    if (!rootVerificationCompleted || !expand_node(0, true) || tree[0].children.empty()) {
        bool completed = true;
        NNUE::Accumulator accumulator;
        NNUE::BerserkNNUE::instance().refresh(accumulator, root);
        root_best_ = MOVE_NONE;
        result.score = negamax(root, std::max(2, GlobalOptions.tactical_depth),
                               -VALUE_INFINITE, VALUE_INFINITE, 0, completed, false, &accumulator);
        result.best_move = root_best_;
        if (!completed || result.best_move == MOVE_NONE) {
            Value best = -VALUE_INFINITE;
            for (Move move : rootLegal) {
                Position child = root;
                StateInfo state;
                if (!child.make_move(move, state)) continue;
                const Value score = -safe_eval(child);
                if (score > best) { best = score; result.best_move = move; }
            }
            result.score = best == -VALUE_INFINITE ? VALUE_DRAW : best;
        }
        result.depth = completed ? std::max(2, GlobalOptions.tactical_depth) : 1;
        result.nodes = nodes_;
        result.pv = move_to_uci(result.best_move);
        result.lower = result.upper = result.score;
        result.confidence = completed ? 85 : 40;
        result.score_source = completed ? "fallback-ab" : "fallback-primary";
        return result;
    }

    auto root_move_interval = [&](int edgeIdx) {
        return negate_corridor(tree[tree[0].children[edgeIdx].child].interval);
    };

    auto budget_leader = [&]() {
        int best = 0;
        double score = -1e100;
        for (int edgeIdx = 0; edgeIdx < int(tree[0].children.size()); ++edgeIdx) {
            const ProofEdge& edge = tree[0].children[edgeIdx];
            const double s = move_decision(tree[edge.child], edge);
            if (s > score) { score = s; best = edgeIdx; }
        }
        return best;
    };

    auto proof_leader = [&]() {
        int best = 0;
        double score = -1e100;
        for (int edgeIdx = 0; edgeIdx < int(tree[0].children.size()); ++edgeIdx) {
            const ProofEdge& edge = tree[0].children[edgeIdx];
            const Corridor c = root_move_interval(edgeIdx);
            const double s = double(c.lower) + 28.0 * edge.prior - 0.04 * c.width();
            if (s > score) { score = s; best = edgeIdx; }
        }
        return best;
    };

    auto challenger_upper = [&](int leader) {
        int upper = -VALUE_INFINITE;
        for (int edgeIdx = 0; edgeIdx < int(tree[0].children.size()); ++edgeIdx)
            if (edgeIdx != leader) upper = std::max(upper, int(root_move_interval(edgeIdx).upper));
        return upper;
    };

    auto refinable = [&](int idx) -> bool {
        const ProofNode& node = tree[idx];
        return !node.terminal && !node.saturated;
    };

    auto select_frontier = [&](int startEdge, ProofObjective objective) {
        int parentIdx = 0;
        int edgeIdx = startEdge;
        ProofEdge* selected = &tree[parentIdx].children[edgeIdx];
        ++selected->visits;
        int idx = selected->child;
        ++tree[idx].visits;
        while (tree[idx].expansion != ExpansionState::Unexpanded && !tree[idx].children.empty()) {
            if (tree[idx].expansion != ExpansionState::Full && (tree[idx].visits % 3 == 0)) break;
            int chosenEdge = -1;
            double bestScore = -1e100;
            const double total = std::max(1, tree[idx].visits + 1);
            for (int candidateEdge = 0; candidateEdge < int(tree[idx].children.size()); ++candidateEdge) {
                const ProofEdge& edge = tree[idx].children[candidateEdge];
                if (!refinable(edge.child)) continue;
                const ProofNode& child = tree[edge.child];
                const Corridor move = negate_corridor(child.interval);
                const double explore = edge.prior * std::sqrt(total) / (1.0 + edge.visits);
                const double score = objective == ProofObjective::TightenUpper
                    ? double(move.upper) + 0.34 * move.width() + 430.0 * explore
                    : 0.62 * double(move.upper) + 0.38 * double(move.lower)
                        + 0.24 * move.width() + 500.0 * explore;
                if (score > bestScore) { bestScore = score; chosenEdge = candidateEdge; }
            }
            if (chosenEdge < 0) break;
            selected = &tree[idx].children[chosenEdge];
            ++selected->visits;
            idx = selected->child;
            ++tree[idx].visits;
            objective = objective == ProofObjective::TightenUpper
                ? ProofObjective::TightenLower : ProofObjective::TightenUpper;
        }
        return idx;
    };

    auto verify_leaf = [&](int idx) -> bool {
        ProofNode& node = tree[idx];
        if (node.terminal) return false;
        int depth = std::clamp(currentVerificationDepth, 1, limits.infinite ? 64 : 8);
        if (node.danger >= 100 || node.noisy) depth = std::min(limits.infinite ? 64 : 8, depth + 1);
        if (node.tacticalVerified && node.verifiedRevision == node.localRevision
                && node.verifiedDepth >= depth)
            return false;
        bool completed = true;
        Position& position = positionArena[node.positionId];
        const Value score = negamax(position, depth, -VALUE_INFINITE, VALUE_INFINITE,
                                    node.ply, completed, true, nullptr);
        if (!completed) {
            ++stats.abortedVerifications;
            return false;
        }
        node.tactical = score;
        node.tacticalVerified = true;
        stats.maxVerificationPly = std::max(stats.maxVerificationPly, node.ply + depth);
        if (node.expansion == ExpansionState::Unexpanded) {
            rebuild_leaf(idx);
        } else {
            const Corridor previous = node.localInterval;
            const int margin = 18 + node.danger / 5;
            node.localInterval = Corridor{
                clamp_non_mate(std::min(int(previous.lower), int(score) - margin)),
                clamp_non_mate(std::max(int(previous.upper), int(score) + margin))};
            if (node.localInterval.lower != previous.lower || node.localInterval.upper != previous.upper)
                ++node.localRevision;
        }
        node.verifiedRevision = node.localRevision;
        node.verifiedDepth = depth;
        ++stats.tacticalRefinements;
        backup_ancestors(idx);
        return true;
    };

    auto choose_root_target = [&](int leader) -> std::pair<int, ProofObjective> {
        const Corridor best = root_move_interval(leader);
        const int chUpper = challenger_upper(leader);
        const int totalVisits = std::accumulate(tree[0].children.begin(), tree[0].children.end(), 0,
            [&](int sum, const ProofEdge& edge) { return sum + edge.visits; });
        int chosen = -1;
        double bestPriority = -1e100;
        ProofObjective objective = ProofObjective::TightenLower;
        for (int edgeIdx = 0; edgeIdx < int(tree[0].children.size()); ++edgeIdx) {
            const ProofEdge& edge = tree[0].children[edgeIdx];
            if (!refinable(edge.child)) continue;
            const ProofNode& child = tree[edge.child];
            const Corridor move = root_move_interval(edgeIdx);
            const int pressure = edgeIdx == leader
                ? std::max(0, chUpper - int(move.lower) + 12)
                : std::max(0, int(move.upper) - int(best.lower) + 12);
            const double explore = edge.prior * std::sqrt(double(totalVisits + 1)) / (1.0 + edge.visits);
            const double priority = 1.65 * pressure + 0.48 * move.width() + 620.0 * explore
                                  + 0.28 * child.danger + (child.hasLeela ? 0.0 : 90.0)
                                  - 9.0 * std::log1p(double(edge.visits));
            if (priority > bestPriority) {
                bestPriority = priority;
                chosen = edgeIdx;
                // Root move interval is the negation of the child interval.
                objective = edgeIdx == leader ? ProofObjective::TightenUpper
                                               : ProofObjective::TightenLower;
            }
        }
        return {chosen, objective};
    };

    auto tree_pv = [&](int rootEdge) {
        std::ostringstream pv;
        int parentIdx = 0;
        int edgeIdx = rootEdge;
        int guard = 0;
        while (parentIdx >= 0 && parentIdx < int(tree.size())
               && edgeIdx >= 0 && edgeIdx < int(tree[parentIdx].children.size()) && guard++ < 64) {
            const ProofEdge& edge = tree[parentIdx].children[edgeIdx];
            if (pv.tellp() > 0) pv << ' ';
            pv << move_to_uci(edge.move);
            const int childIdx = edge.child;
            if (tree[childIdx].expansion == ExpansionState::Unexpanded
                    || tree[childIdx].bestEdge < 0) break;
            parentIdx = childIdx;
            edgeIdx = tree[childIdx].bestEdge;
        }
        return pv.str();
    };

    auto emit_progress = [&](int chosen, const char* phase) {
        if (!GlobalOptions.gui_progress || chosen < 0) return;
        const ProofEdge& rootEdge = tree[0].children[chosen];
        const Corridor c = root_move_interval(chosen);
        const std::uint64_t ms = std::max<std::uint64_t>(1, elapsed_ms());
        std::ostringstream out;
        out << "info depth " << std::max(1, tree[0].subtreeDepth)
            << " seldepth " << std::max(stats.maxTreePly, stats.maxVerificationPly)
            << " score cp " << c.center()
            << " time " << ms
            << " nodes " << nodes_
            << " nps " << nodes_ * 1000ull / ms
            << " string ModernBStar frontier phase " << phase
            << " best " << move_to_uci(rootEdge.move)
            << " L " << c.lower << " U " << c.upper << " W " << c.width()
            << " prior " << std::fixed << std::setprecision(3) << rootEdge.prior
            << std::defaultfloat
            << " tree " << tree.size()
            << " epoch " << currentEpoch
            << " activebudget " << activeNodeBudget
            << " verifydepth " << currentVerificationDepth
            << " expansions " << stats.frontierExpansions
            << " iterations " << stats.frontierIterations
            << " ld2budget " << leelaBudget << "/" << leelaBudgetStart
            << " pv " << tree_pv(chosen) << "\n";
        emit_info(out.str());
    };

    int previousLeader = proof_leader();
    bool hardProof = false;
    bool practicalProof = false;
    auto lastProgress = Clock::now();
    emit_progress(budget_leader(), "root");

    auto advance_epoch = [&]() {
        if (activeNodeBudget < nodeLimit) {
            activeNodeBudget = std::min(nodeLimit, activeNodeBudget * 2);
            wideningTarget = std::min(256, wideningTarget * 2);
            currentVerificationDepth = std::min(limits.infinite ? 64 : 8,
                                                currentVerificationDepth + 1);
            ++currentEpoch;
            return true;
        }
        if (limits.infinite && currentVerificationDepth < 64) {
            currentVerificationDepth = std::min(64, currentVerificationDepth + 1);
            ++currentEpoch;
            return true;
        }
        return false;
    };

    while (!should_stop()) {
        const int leader = proof_leader();
        const ProofEdge& leaderEdge = tree[0].children[leader];
        const ProofNode& leaderNode = tree[leaderEdge.child];
        if (leader != previousLeader) {
            ++stats.leaderChanges;
            previousLeader = leader;
        }
        const Corridor best = root_move_interval(leader);
        const int chUpper = challenger_upper(leader);
        const bool evidence = stats.frontierIterations >= GlobalOptions.min_proof_expansions
                           && (!leelaAvailable || leaderNode.hasLeela)
                           && leaderEdge.visits >= 2;
        hardProof = evidence && best.lower > chUpper;
        practicalProof = evidence && best.lower + GlobalOptions.practical_margin > chUpper
                       && best.width() <= 120 && stats.leaderChanges <= 4;
        if (best.lower >= VALUE_MATE_IN_MAX_PLY) break;

        const bool minimumTimeUsed = limits.movetime_ms <= 0
            || elapsed_ms() >= std::uint64_t(std::max(1, limits.movetime_ms * 35 / 100));
        if (!limits.infinite && GlobalOptions.heuristic_early_stop
                && minimumTimeUsed && (hardProof || practicalProof)) break;

        auto [target, objective] = choose_root_target(leader);
        if (target < 0) {
            if (advance_epoch()) {
                (void)verify_leaf(tree[0].children[leader].child);
                emit_progress(budget_leader(), "epoch");
                continue;
            }
            if (!limits.infinite) break;
            // A finite proof sweep can exhaust its configured tree. In UCI
            // analysis mode that is not permission to emit bestmove: retain
            // the completed result and wait for an explicit `stop`.
            const auto now = Clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProgress).count()
                    >= GlobalOptions.gui_progress_interval_ms) {
                emit_progress(budget_leader(), "frontier-exhausted");
                lastProgress = now;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        const int leaf = select_frontier(target, objective);
        bool changed = false;
        const bool needsTacticalSanitizer = (!tree[leaf].tacticalVerified
                || tree[leaf].verifiedRevision != tree[leaf].localRevision
                || tree[leaf].verifiedDepth < currentVerificationDepth);
        if (needsTacticalSanitizer)
            changed = verify_leaf(leaf);
        if (!changed && !tree[leaf].hasLeela && leelaAvailable
                && leelaBudget > 0 && enough_time_for_leela()) {
            // A single-threaded best-first driver has no independent producers
            // to wait on, so its scheduler fills immediately from compatible
            // sibling/root contenders and flushes early at the deadline. This
            // still gives the backend one true tensor batch rather than N
            // scalar calls.
            std::vector<int> scheduled{leaf};
            const std::size_t batchSize = std::size_t(
                std::clamp(GlobalOptions.ld2_batch_size, 1, 32));
            auto schedule_candidate = [&](int candidate) {
                if (scheduled.size() >= batchSize || candidate < 0
                        || candidate >= int(tree.size())) return;
                const ProofNode& node = tree[candidate];
                if (node.hasLeela || node.terminal || node.saturated) return;
                if (std::find(scheduled.begin(), scheduled.end(), candidate) == scheduled.end())
                    scheduled.push_back(candidate);
            };
            for (int parent : tree[leaf].parents) {
                for (const ProofEdge& edge : tree[parent].children) schedule_candidate(edge.child);
                if (scheduled.size() >= batchSize) break;
            }
            for (const ProofEdge& edge : tree[0].children) schedule_candidate(edge.child);

            const auto updated = probe_leela_batch(scheduled);
            for (const int candidate : updated) {
                rebuild_leaf(candidate);
                backup_ancestors(candidate);
                ++stats.leelaRefinements;
                if (candidate == leaf) changed = true;
            }
        }
        if (!changed && !should_stop())
            changed = expand_node(leaf, false);
        if (!changed && tree[leaf].ply >= maxTreeDepth) {
            tree[leaf].saturated = true;
            backup_ancestors(leaf);
        }
        ++stats.frontierIterations;

        const auto now = Clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProgress).count()
                >= GlobalOptions.gui_progress_interval_ms) {
            emit_progress(budget_leader(), "search");
            lastProgress = now;
        }
        if (int(tree.size()) >= activeNodeBudget) {
            if (advance_epoch()) {
                emit_progress(budget_leader(), "epoch");
                continue;
            }
            if (!limits.infinite) break;
            const auto budgetNow = Clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(budgetNow - lastProgress).count()
                    >= GlobalOptions.gui_progress_interval_ms) {
                emit_progress(budget_leader(), "node-budget");
                lastProgress = budgetNow;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    const int proofBest = proof_leader();
    const Corridor proofBestInterval = root_move_interval(proofBest);
    const int proofChUpper = challenger_upper(proofBest);
    const ProofEdge& proofBestEdge = tree[0].children[proofBest];
    const ProofNode& proofBestNode = tree[proofBestEdge.child];
    const bool evidence = stats.frontierIterations >= GlobalOptions.min_proof_expansions
                       && (!leelaAvailable || proofBestNode.hasLeela)
                       && proofBestEdge.visits >= 2;
    hardProof = evidence && proofBestInterval.lower > proofChUpper;
    practicalProof = evidence && proofBestInterval.lower + GlobalOptions.practical_margin > proofChUpper
                   && proofBestInterval.width() <= 120 && stats.leaderChanges <= 4;
    const int finalChild = hardProof || practicalProof ? proofBest : budget_leader();
    const Corridor finalInterval = root_move_interval(finalChild);
    const int decisionChUpper = challenger_upper(finalChild);
    const ProofEdge& finalEdge = tree[0].children[finalChild];
    const ProofNode& finalNode = tree[finalEdge.child];

    result.best_move = finalEdge.move;
    result.lower = finalInterval.lower;
    result.upper = finalInterval.upper;
    result.confidence = finalNode.confidence;
    if (finalInterval.lower == finalInterval.upper) {
        result.score = finalInterval.center();
        result.score_source = "exact";
        result.confidence = 100;
    } else if (finalNode.tacticalVerified && finalNode.tactical != VALUE_NONE) {
        result.score = Value(std::clamp(-int(finalNode.tactical), int(finalInterval.lower), int(finalInterval.upper)));
        result.score_source = "tactical";
    } else if (finalNode.hasLeela) {
        result.score = finalInterval.center();
        result.score_source = "fused";
    } else {
        result.score = Value(std::clamp(-int(finalNode.primary), int(finalInterval.lower), int(finalInterval.upper)));
        result.score_source = NNUE::BerserkNNUE::instance().loaded() ? "berserk-nnue" : "hce";
    }
    result.depth = std::max(1, tree[0].subtreeDepth);
    result.seldepth = std::max({result.depth, stats.maxTreePly, stats.maxVerificationPly});
    result.proof_iterations = stats.frontierIterations;
    result.proof_effort = stats.frontierIterations + stats.tacticalRefinements * GlobalOptions.tactical_depth;
    result.frontier_expansions = stats.frontierExpansions;
    result.tree_nodes = int(tree.size());
    result.hard_proof = hardProof;
    result.practical_proof = practicalProof;
    result.nodes = nodes_;
    result.time_ms = elapsed_ms();
    result.pv = tree_pv(finalChild);
    root_best_ = result.best_move;

    tt_.save_corridor(tt_domain_key(proof_context_hash(root), TTDomain::ModernCorridor), finalInterval.lower, finalInterval.upper, result.depth,
                      move_to_tt(result.best_move), 0, hardProof || practicalProof,
                      -finalNode.primary, true);

    if (GlobalOptions.debug_modern_bstar) {
        std::ostringstream debug;
        debug << "info string mbstar config root_berserk=" << tree[0].primary
              << " root_leela=";
        if (tree[0].hasLeela) debug << tree[0].leela;
        else debug << "na";
        debug << " root_fused=" << tree[0].fused
              << " leela_value_weight_mg=" << GlobalOptions.leela_value_weight
              << " leela_value_weight_eg=" << GlobalOptions.leela_endgame_value_weight
              << " leela_policy_weight_mg=" << GlobalOptions.leela_policy_weight
              << " leela_policy_weight_eg=" << GlobalOptions.leela_endgame_policy_weight
              << " tactical_depth=" << GlobalOptions.tactical_depth << "\n";
        for (int edgeIdx = 0; edgeIdx < int(tree[0].children.size()); ++edgeIdx) {
            const ProofEdge& edge = tree[0].children[edgeIdx];
            const ProofNode& child = tree[edge.child];
            const Corridor c = root_move_interval(edgeIdx);
            debug << "info string mbstar root move=" << move_to_uci(edge.move)
                  << " L=" << c.lower << " U=" << c.upper << " C=" << c.center()
                  << " W=" << c.width() << " b=" << -child.primary << " l=";
            if (child.hasLeela) debug << -child.leela;
            else debug << "na";
            debug << " tv=";
            if (child.tacticalVerified) debug << -child.tactical;
            else debug << "na";
            debug << " fused=" << -child.fused
                  << " pol=" << std::fixed << std::setprecision(4) << edge.prior
                  << " risk=" << child.danger << " conf=" << child.confidence
                  << " visits=" << edge.visits << " depth=" << child.subtreeDepth
                  << " leela=" << bool01(child.hasLeela)
                  << " tact=" << bool01(child.tacticalVerified)
                  << " expansion=" << (child.expansion == ExpansionState::Full ? "full"
                                         : child.expansion == ExpansionState::Partial ? "partial"
                                                                                     : "unexpanded") << "\n";
        }
        debug << "info string mbstar decision best=" << move_to_uci(finalEdge.move)
              << " proof_leader=" << move_to_uci(proofBestEdge.move)
              << " heuristic_strict=" << bool01(hardProof)
              << " heuristic_practical=" << bool01(practicalProof)
              << " decision_challengerU=" << decisionChUpper
              << " proof_challengerU=" << proofChUpper
              << " reason=" << (hardProof ? "heuristic_interval_separation"
                                            : practicalProof ? "heuristic_practical_separation"
                                                             : "budget_decision") << "\n";
        result.debug_info = debug.str();
    }

    const auto nn = NNUE::BerserkNNUE::instance().status();
    const auto ld = Leela::RawBackend::instance().status();
    const std::uint64_t nnEvals = nn.evals - nnStart.evals;
    const std::uint64_t nnNs = nn.total_ns - nnStart.total_ns;
    const std::uint64_t ldProbes = ld.probes - ldStart.probes;
    const std::uint64_t ldUs = ld.total_us - ldStart.total_us;
    const std::uint64_t ldBatchCalls = ld.batch_calls - ldStart.batch_calls;
    const std::uint64_t ldBatchPositions = ld.batch_positions - ldStart.batch_positions;
    const int rootPhase = material_phase(root);
    const int rootValueWeight = tapered_ld2_weight(root, GlobalOptions.leela_value_weight,
                                                    GlobalOptions.leela_endgame_value_weight, 80);
    const int rootPolicyWeight = tapered_ld2_weight(root, GlobalOptions.leela_policy_weight,
                                                     GlobalOptions.leela_endgame_policy_weight, 90);
    const auto partialNodes = std::count_if(tree.begin(), tree.end(), [](const ProofNode& node) {
        return node.expansion == ExpansionState::Partial;
    });
    std::ostringstream info;
    info << (hardProof ? "heuristic_strict" : practicalProof ? "heuristic_practical" : "budget")
         << " best " << move_to_uci(result.best_move)
         << " lower " << finalInterval.lower << " upper " << finalInterval.upper
         << " width " << finalInterval.width() << " maxChallengerUpper " << decisionChUpper
         << " tree_nodes " << tree.size()
         << " partial_nodes " << partialNodes
         << " transposition_hits " << stats.transpositionHits
         << " tt_order_hits " << stats.ttOrderHits
         << " syzygy_leaves " << stats.syzygyLeaves
         << " frontier_expansions " << stats.frontierExpansions
         << " frontier_iterations " << stats.frontierIterations
         << " leela_refinements " << stats.leelaRefinements
         << " tactical_refinements " << stats.tacticalRefinements
         << " aborted_verifications " << stats.abortedVerifications
         << " leader_changes " << stats.leaderChanges
         << " material_phase " << rootPhase
         << " leela_value_weight " << rootValueWeight
         << " leela_policy_weight " << rootPolicyWeight
         << " heuristic_early_stop " << bool01(GlobalOptions.heuristic_early_stop)
         << " nnue_evals " << nnEvals
         << " nnue_avg_ns " << (nnEvals ? nnNs / nnEvals : 0)
         << " ld2_probes " << ldProbes
         << " ld2_avg_us " << (ldProbes ? ldUs / ldProbes : 0)
         << " ld2_batch_calls " << ldBatchCalls
         << " ld2_batch_positions " << ldBatchPositions
         << " ld2_batch_workers " << ld.batch_workers
         << " ld2_budget " << leelaBudget << "/" << leelaBudgetStart;
    result.search_info = info.str();
    return result;
}

Result Searcher::think_classic_ab(Position& root, const Limits& limits) {
    limits_ = limits;
    start_ = Clock::now();
    if (!limits.ponder) control_.set_deadline_after(limits.movetime_ms);
    ponderhit_active_ = false;
    nodes_ = 0;
    root_best_ = MOVE_NONE;
    tt_.new_search();

    Result result;
    MoveList legal;
    generate_legal(root, legal);
    restrict_root_moves(legal, limits);
    if (legal.size == 0) {
        result.score = root.in_check(root.side) ? mated_in(0) : VALUE_DRAW;
        return result;
    }
    result.best_move = legal.moves[0];
    result.pv = move_to_uci(result.best_move);

    NNUE::Accumulator rootAccumulator;
    NNUE::BerserkNNUE::instance().refresh(rootAccumulator, root);

    const int maxDepth = std::max(1, limits.depth);
    for (int depth = 1; depth <= maxDepth && !should_stop(); ++depth) {
        bool completed = true;
        root_best_ = MOVE_NONE;
        const Value score = negamax(root, depth, -VALUE_INFINITE, VALUE_INFINITE,
                                    0, completed, false, &rootAccumulator);
        if (!completed) break;
        result.score = score;
        result.depth = depth;
        result.seldepth = depth;
        if (root_best_ != MOVE_NONE) result.best_move = root_best_;
        result.pv = extract_pv(root, depth);
        if (result.pv.empty()) result.pv = move_to_uci(result.best_move);
    }
    // A finite iterative-deepening sweep is not allowed to turn `go infinite`
    // or `go ponder` into an unsolicited bestmove. ClassicAB has no native
    // progress driver, so it waits cheaply for stop/ponderhit once its current
    // depth cap is exhausted.
    while (limits.infinite && !should_stop())
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    result.requested_depth = maxDepth;
    result.nodes = nodes_;
    result.time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_).count();
    result.search_info = "ClassicAB iterative tactical baseline";
    result.lower = result.upper = result.score;
    result.confidence = 100;
    result.score_source = "classic-ab";
    const auto& nnStatus = NNUE::BerserkNNUE::instance().status();
    result.degraded = nnStatus.enabled && !nnStatus.loaded;
    return result;
}

Result Searcher::think(Position& root, const Limits& limits) {
    // `tb_probe_root` is DTZ-aware and root-only. It is therefore the sole
    // source allowed to end a finite root search immediately. In analysis or
    // `searchmoves` mode we retain normal UCI/search semantics.
    if (!limits.infinite && limits.searchmoves.empty() && !draw_by_rule(root, true)) {
        const auto start = Clock::now();
        if (const auto tb = Syzygy::Backend::instance().probe_root(root)) {
            Result result;
            result.best_move = tb->best_move;
            result.score = syzygy_root_score(tb->wdl);
            result.depth = result.seldepth = 1;
            result.nodes = 1;
            result.time_ms = std::uint64_t(std::max<long long>(0,
                std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count()));
            result.pv = move_to_uci(result.best_move);
            result.search_info = "Syzygy exact root wdl " + std::to_string(int(tb->wdl))
                               + " dtz " + std::to_string(tb->dtz);
            result.lower = result.upper = result.score;
            result.confidence = 100;
            result.score_source = "syzygy";
            return result;
        }
    }
    return GlobalOptions.modern_bstar ? think_modern_bstar(root, limits) : think_classic_ab(root, limits);
}

std::string Searcher::extract_pv(Position& root, int max_depth) {
    Position pos = root;
    std::ostringstream out;
    for (int ply = 0; ply < max_depth; ++ply) {
        auto probe = tt_.probe(pos.key, ply);
        if (!probe.hit || !probe.data.has_move) break;
        Move move = resolve_tt_move(pos, probe.data.move);
        if (move == MOVE_NONE) break;
        if (ply) out << ' ';
        out << move_to_uci(move);
        StateInfo st;
        if (!pos.make_move(move, st)) break;
    }
    return out.str();
}

} // namespace Scarlet::Search

#include "leela_raw.hpp"
#include "dag_backup.hpp"
#include "movegen.hpp"
#include "nnue_berserk.hpp"
#include "search.hpp"
#include "tt.hpp"
#include "neural/tables/policy_map.h"

#include <array>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>

namespace {

using namespace Scarlet;

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
}

int special_move_index_for_d7(int direction, PieceType promotion) {
    // In LC0 kMoveStrs the promotion tail starts at 1792, is ordered by
    // source file, destination left/forward/right, then q/r/b. Before d7
    // there are 6 (a-file) + 9 (b) + 9 (c) entries.
    int pieceOffset = -1;
    if (promotion == QUEEN) pieceOffset = 0;
    if (promotion == ROOK) pieceOffset = 1;
    if (promotion == BISHOP) pieceOffset = 2;
    if (pieceOffset < 0) fail("non-special promotion passed to LC0 reference");
    return 1792 + 24 + direction * 3 + pieceOffset;
}

int reference_plane(int direction, PieceType promotion) {
    if (promotion == KNIGHT) {
        // White-oriented promotion from d7: left/forward/right are the
        // ordinary NW/N/NE one-step queen-ray planes.
        constexpr std::array<int, 3> normalPlanes{49, 0, 7};
        const int expectedFlat = normalPlanes[direction] * 64 + 51; // d7
        if (lczero::kConvPolicyMap[expectedFlat] < 0)
            fail("LC0 reference rejected normal promotion plane");
        return normalPlanes[direction];
    }

    const int expectedMove = special_move_index_for_d7(direction, promotion);
    for (int flat = 64 * 64; flat < 73 * 64; ++flat)
        if (lczero::kConvPolicyMap[flat] == expectedMove) return flat / 64;
    fail("LC0 reference promotion move not found");
}

void test_promotion_policy_mapping() {
    constexpr std::array<PieceType, 4> promotions{KNIGHT, BISHOP, ROOK, QUEEN};
    int checked = 0;
    for (Color side : {WHITE, BLACK}) {
        Position pos;
        const char* fen = side == WHITE
            ? "4k3/3P4/8/8/8/8/8/4K3 w - - 0 1"
            : "4k3/8/8/8/8/8/3p4/4K3 b - - 0 1";
        if (!pos.set_fen(fen)) fail("promotion test FEN rejected");
        const int from = side == WHITE ? 51 : 11; // d7 / d2
        for (int direction = 0; direction < 3; ++direction) {
            const int fileDelta = direction - 1;
            const int to = from + (side == WHITE ? 8 : -8) + fileDelta;
            for (PieceType promotion : promotions) {
                const Move move = make_move(from, to, MF_PROMOTION, promotion);
                const int actual = Leela::RawBackend::canonical_policy_plane(pos, move);
                const int expected = reference_plane(direction, promotion);
                if (actual != expected) {
                    fail("promotion plane mismatch for " + move_to_uci(move)
                         + ": expected " + std::to_string(expected)
                         + ", got " + std::to_string(actual));
                }
                ++checked;
            }
        }
    }
    if (checked != 24) fail("promotion parity did not cover 24 cases");
}

void play(Position& position, const char* uci) {
    MoveList legal;
    generate_legal(position, legal);
    for (Move move : legal) {
        if (move_to_uci(move) != uci) continue;
        StateInfo state;
        if (!position.make_move(move, state)) break;
        return;
    }
    fail(std::string("cannot play test move ") + uci);
}

void compare_probe(const Leela::RawProbeResult& expected,
                   const Leela::RawProbeResult& actual, float tolerance) {
    for (int i = 0; i < 3; ++i)
        if (std::abs(expected.wdl[i] - actual.wdl[i]) > tolerance)
            fail("batched WDL differs from B=1");
    if (expected.policy.size() != actual.policy.size()) fail("batched policy size differs");
    for (const auto& entry : expected.policy) {
        bool found = false;
        for (const auto& candidate : actual.policy) {
            if (!same_move_core(candidate.move, move_to_tt(entry.move))) continue;
            if (candidate.plane != entry.plane || std::abs(candidate.prior - entry.prior) > tolerance)
                fail("batched policy differs from B=1");
            found = true;
            break;
        }
        if (!found) fail("batched policy move missing");
    }
}

void test_ld2_batch_and_lru() {
    auto& backend = Leela::RawBackend::instance();
    if (!backend.try_load_default()) return;
    backend.set_threads(4);
    backend.set_batch_workers(4);

    std::vector<Position> positions(8);
    for (auto& position : positions)
        if (!position.set_fen(START_FEN)) fail("start FEN rejected");
    play(positions[1], "e2e4");
    play(positions[2], "d2d4");
    play(positions[3], "g1f3");
    play(positions[4], "c2c4");
    play(positions[5], "e2e4"); play(positions[5], "e7e5");
    play(positions[6], "d2d4"); play(positions[6], "d7d5");
    play(positions[7], "g1f3"); play(positions[7], "g8f6");

    std::vector<Leela::RawProbeResult> scalar;
    for (const auto& position : positions) {
        auto result = backend.scalar_reference_probe(position, VALUE_ZERO);
        if (!result || !result->valid) fail("B=1 LD2 probe failed");
        scalar.push_back(std::move(*result));
    }
    for (int batchSize : {1, 2, 4, 8, 32}) {
        std::vector<Leela::RawProbeRequest> requests;
        for (int i = 0; i < batchSize; ++i)
            requests.push_back(Leela::RawProbeRequest{positions[i % positions.size()], VALUE_ZERO});
        backend.clear_cache();
        const auto results = backend.probe_batch(requests);
        if (results.size() != requests.size()) fail("LD2 batch result size mismatch");
        for (int i = 0; i < batchSize; ++i) {
            if (!results[i] || !results[i]->valid) fail("LD2 batch probe failed");
            compare_probe(scalar[i % scalar.size()], *results[i], 1e-5f);
        }
    }
    if (backend.status().workspace_reallocations != 0)
        fail("LD2 tensor workspace reallocated after load warmup");

    backend.set_cache_capacity(2);
    backend.clear_cache();
    (void)backend.probe(positions[0], VALUE_ZERO);
    (void)backend.probe(positions[1], VALUE_ZERO);
    (void)backend.probe(positions[0], VALUE_ZERO); // promote to MRU
    (void)backend.probe(positions[2], VALUE_ZERO); // must evict positions[1]
    if (!backend.cached_probe(positions[0]) || !backend.cached_probe(positions[2]))
        fail("LRU evicted an MRU entry");
    if (backend.cached_probe(positions[1])) fail("LRU retained the wrong victim");
    backend.set_cache_capacity(16384);

    // Same board, side, castling and en-passant, but a different 8-frame
    // history must not alias the network-input cache.
    Position repeatedBoard;
    if (!repeatedBoard.set_fen(START_FEN)) fail("history cache FEN rejected");
    play(repeatedBoard, "g1f3"); play(repeatedBoard, "g8f6");
    play(repeatedBoard, "f3g1"); play(repeatedBoard, "f6g8");
    if (repeatedBoard.key != positions[0].key) fail("test did not recreate the same board");
    backend.clear_cache();
    (void)backend.probe(positions[0], VALUE_ZERO);
    if (backend.cached_probe(repeatedBoard))
        fail("network cache mixed equal boards with different history");
}

void test_exact_perft() {
    struct Case { const char* fen; int depth; std::uint64_t expected; };
    constexpr std::array<Case, 2> cases{{
        {START_FEN, 5, 4865609ULL},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
         4, 4085603ULL}
    }};
    for (const auto& test : cases) {
        Position position;
        if (!position.set_fen(test.fen)) fail("perft FEN rejected");
        const auto actual = perft(position, test.depth);
        if (actual != test.expected)
            fail("perft mismatch: expected " + std::to_string(test.expected)
                 + ", got " + std::to_string(actual));
    }
}

void test_nnue_incremental_parity() {
    auto& nnue = NNUE::BerserkNNUE::instance();
    if (!nnue.try_load_default()) return;
    struct Case { const char* fen; const char* move; };
    constexpr std::array<Case, 7> cases{{
        {START_FEN, "e2e4"},
        {"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"},
        {"4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"},
        {"7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8q"},
        {"7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8r"},
        {"7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8b"},
        {"7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8n"}
    }};
    for (const auto& test : cases) {
        Position position;
        if (!position.set_fen(test.fen)) fail("NNUE parity FEN rejected");
        NNUE::Accumulator accumulator;
        nnue.refresh(accumulator, position);
        MoveList legal;
        generate_legal(position, legal);
        Move selected = MOVE_NONE;
        for (Move move : legal)
            if (move_to_uci(move) == test.move) { selected = move; break; }
        if (selected == MOVE_NONE) fail(std::string("NNUE parity move unavailable: ") + test.move);
        StateInfo state;
        if (!position.make_move(selected, state)) fail("NNUE parity make_move failed");
        nnue.apply_move(accumulator, position, selected, state);
        if (!nnue.matches_full_rebuild(accumulator, position))
            fail(std::string("incremental NNUE mismatch after ") + test.move);
    }
}

void test_tt_resize_strong_guarantee() {
    using namespace scarlet::tt;
    TranspositionTable table(Config{1, EvalCache::Sidecar, false, true});
    constexpr Key key = 0x123456789ABCDEF0ULL;
    table.save_classic(key, 123, Bound::Exact, 6, 42, 17, 0, false, true);
    const auto oldClusters = table.cluster_count();
    auto require_old_entry = [&] {
        const auto probe = table.probe(key, 0);
        if (!probe.hit || !probe.data.has_classic_score || probe.data.lower != 123
                || table.cluster_count() != oldClusters)
            fail("TT state changed after failed resize");
    };

    for (int failurePoint : {0, 1}) {
        detail::set_allocation_failure_countdown(failurePoint);
        bool threw = false;
        try {
            table.resize(Config{2, EvalCache::Sidecar, false, true});
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        detail::set_allocation_failure_countdown(-1);
        if (!threw) fail("TT fault injection did not fail resize");
        require_old_entry();
    }
}

void test_network_rejects_corruption() {
    auto& backend = Leela::RawBackend::instance();
    if (!backend.try_load_default()) return;
    const auto validPath = backend.status().weights_path;
    const auto temp = std::filesystem::temp_directory_path() / "scarlet-invalid-ld2.pb";
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        const std::array<unsigned char, 8> bad{0, 1, 2, 3, 4, 5, 6, 7};
        out.write(reinterpret_cast<const char*>(bad.data()), bad.size());
    }
    if (backend.load(temp.string())) fail("LD2 loader accepted invalid magic/truncation");
    std::error_code ignored;
    std::filesystem::remove(temp, ignored);

    std::ifstream valid(validPath, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(valid)),
                            std::istreambuf_iterator<char>());
    if (bytes.size() < 64) fail("bundled LD2 unexpectedly small");
    const auto truncated = std::filesystem::temp_directory_path() / "scarlet-truncated-ld2.pb";
    {
        std::ofstream out(truncated, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), std::streamsize(bytes.size() / 2));
    }
    if (backend.load(truncated.string())) fail("LD2 loader accepted truncated protobuf");
    std::filesystem::remove(truncated, ignored);

    std::string wrongFormat = bytes;
    const std::array<unsigned char, 8> signature{0x1a, 0x06, 0x08, 0x00, 0x10, 0x15, 0x18, 0x00};
    const auto signatureAt = std::search(
        wrongFormat.begin(), wrongFormat.end(), signature.begin(), signature.end(),
        [](char lhs, unsigned char rhs) { return static_cast<unsigned char>(lhs) == rhs; });
    if (signatureAt == wrongFormat.end()) fail("LD2 format signature not found");
    wrongFormat[std::distance(wrongFormat.begin(), signatureAt) + 5] = char(0x7f);
    const auto wrongFormatPath = std::filesystem::temp_directory_path() / "scarlet-wrong-format-policy_value.pb";
    {
        std::ofstream out(wrongFormatPath, std::ios::binary | std::ios::trunc);
        out.write(wrongFormat.data(), std::streamsize(wrongFormat.size()));
    }
    if (backend.load(wrongFormatPath.string())) fail("LD2 loader accepted unsupported format enum");
    std::filesystem::remove(wrongFormatPath, ignored);
    if (!backend.load(validPath)) fail("LD2 loader did not recover after invalid file");
}

void test_dag_backup_order() {
    struct Node { int ply; int value; std::vector<int> children; std::vector<int> parents; };
    // root -> A/B -> shared leaf
    std::vector<Node> dag{
        {0, 0, {1, 2}, {}}, {1, 0, {3}, {0}},
        {1, 0, {3}, {0}}, {2, 7, {}, {1, 2}}};
    std::vector<int> order;
    Search::propagate_dag_change(dag.size(), 3,
        [&](int node) { return dag[node].ply; },
        [&](int node) -> const std::vector<int>& { return dag[node].parents; },
        [&](int node) {
            order.push_back(node);
            if (dag[node].children.empty()) return false;
            int next = 0;
            for (int child : dag[node].children) next += dag[child].value;
            const bool changed = next != dag[node].value;
            dag[node].value = next;
            return changed;
        });
    if (dag[1].value != 7 || dag[2].value != 7 || dag[0].value != 14)
        fail("DAG backup left stale root state");
    const auto rootAt = std::find(order.begin(), order.end(), 0);
    const auto aAt = std::find(order.begin(), order.end(), 1);
    const auto bAt = std::find(order.begin(), order.end(), 2);
    if (rootAt == order.end() || rootAt < aAt || rootAt < bAt)
        fail("DAG root was backed before all changed parents");
}

void test_deterministic_root_mates() {
    struct Case { const char* fen; const char* move; };
    constexpr std::array<Case, 2> cases{{
        {"4rkr1/1p1Rn1pp/p3p2B/4Qp2/8/8/PPq2PPP/3R2K1 w - - 0 1", "e5f6"},
        {"r1b2rk1/pppp2p1/8/3qPN1Q/8/8/P5PP/b1B2R1K w - - 0 1", "f5e7"}
    }};
    const bool previousLd2 = Search::options().use_ld2;
    Search::options().use_ld2 = false;
    Search::Searcher searcher(8);
    for (const auto& test : cases) {
        Position position;
        if (!position.set_fen(test.fen)) fail("mate test FEN rejected");
        Search::Limits limits;
        limits.depth = 16;
        limits.nodes = 500000;
        searcher.reset_stop();
        const auto result = searcher.think(position, limits);
        if (result.best_move == MOVE_NONE || move_to_uci(result.best_move) != test.move)
            fail(std::string("deterministic mate missed: expected ") + test.move
                 + ", got " + (result.best_move == MOVE_NONE ? "0000" : move_to_uci(result.best_move)));
        if (result.score < VALUE_MATE_IN_MAX_PLY)
            fail("deterministic mate did not return a mate score");
    }
    Search::options().use_ld2 = previousLd2;
}

void test_forced_move_is_scored() {
    Position position;
    if (!position.set_fen("7k/8/5K2/6Q1/8/8/8/8 b - - 0 1"))
        fail("forced-move FEN rejected");
    MoveList legal;
    generate_legal(position, legal);
    if (legal.size != 1 || move_to_uci(legal.moves[0]) != "h8h7")
        fail("forced-move test position does not have exactly h8h7");
    Search::Searcher searcher(8);
    Search::Limits limits;
    limits.depth = 8;
    limits.nodes = 100000;
    searcher.reset_stop();
    const auto result = searcher.think(position, limits);
    if (result.best_move == MOVE_NONE || move_to_uci(result.best_move) != "h8h7")
        fail("forced move was not returned");
    if (result.score == VALUE_ZERO || result.score_source == "unknown"
            || result.lower == VALUE_NONE || result.upper == VALUE_NONE)
        fail("forced move returned a synthetic score");
}

} // namespace

int main() {
    Scarlet::init();
    test_promotion_policy_mapping();
    test_ld2_batch_and_lru();
    test_tt_resize_strong_guarantee();
    test_network_rejects_corruption();
    test_dag_backup_order();
    test_exact_perft();
    test_nnue_incremental_parity();
    test_deterministic_root_mates();
    test_forced_move_is_scored();
    std::cout << "ok: promotion-policy parity (24 cases)\n";
    std::cout << "ok: LD2 batched parity and O(1) LRU\n";
    std::cout << "ok: TT strong resize and LD2 corruption rejection\n";
    std::cout << "ok: bottom-up multi-parent DAG backup\n";
    std::cout << "ok: exact startpos and Kiwipete perft\n";
    std::cout << "ok: incremental NNUE normal/castling/en-passant/promotions\n";
    std::cout << "ok: deterministic root mate regression\n";
    std::cout << "ok: forced move has a real score and bounds\n";
    return 0;
}

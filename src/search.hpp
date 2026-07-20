#pragma once

#include "eval.hpp"
#include "tt.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace Scarlet::Search {

struct Options {
    bool modern_bstar = true;
    bool use_ld2 = true;
    bool ld2_root_policy_always = true;
    int ld2_max_root_probes = 10;
    int ld2_max_frontier_probes = 64;
    int ld2_min_policy_permille = 30;
    int ld2_batch_size = 4;
    int proof_node_limit = 65536;
    int tactical_depth = 3;
    int practical_margin = 24;
    int min_proof_expansions = 24;
    int leela_value_weight = 45;          // middlegame percent in fused leaf centre
    int leela_policy_weight = 55;         // middlegame percent in allocator/order blending
    int leela_endgame_value_weight = 15;  // applied at material phase 0
    int leela_endgame_policy_weight = 20; // applied at material phase 0
    // The corridor is heuristic evidence, not a game-theoretic bound. Keep
    // consuming the caller's time by default instead of returning early on it.
    bool heuristic_early_stop = false;
    bool debug_modern_bstar = false;

    // Emit intermediate UCI info lines during Modern-B* proof loops so Arena/CuteChess
    // see changing depth/nodes/nps instead of one frozen final line.
    bool gui_progress = true;
    int gui_progress_interval_ms = 250;
};

Options& options();
void set_modern_bstar_enabled(bool enabled);

class SearchControl {
public:
    void reset() {
        source_ = std::stop_source{};
        ponderhit_requested.store(false, std::memory_order_relaxed);
        nodes.store(0, std::memory_order_relaxed);
        deadline_ns.store(0, std::memory_order_relaxed);
    }
    void request_stop() noexcept { source_.request_stop(); }
    [[nodiscard]] bool stop_requested() const noexcept { return source_.stop_requested(); }
    [[nodiscard]] std::stop_token token() const noexcept { return source_.get_token(); }
    void set_deadline_after(int milliseconds) noexcept {
        if (milliseconds <= 0) {
            deadline_ns.store(0, std::memory_order_relaxed);
            return;
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(milliseconds);
        deadline_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
            deadline.time_since_epoch()).count(), std::memory_order_relaxed);
    }
    [[nodiscard]] bool deadline_expired() const noexcept {
        const auto value = deadline_ns.load(std::memory_order_relaxed);
        if (value <= 0) return false;
        const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        return now >= value;
    }

    std::atomic<bool> ponderhit_requested{false};
    std::atomic<std::uint64_t> nodes{0};
    std::atomic<std::int64_t> deadline_ns{0};

private:
    std::stop_source source_{};
};

struct Limits {
    int depth = 1;
    int movetime_ms = 0;
    std::uint64_t nodes = 0;
    // UCI `go searchmoves`: stable move identities selected by the GUI.
    std::vector<TTMove> searchmoves;

    // UCI analysis mode: `go infinite`. The search must keep producing
    // progress until an external `stop` arrives. This is deliberately separate
    // from movetime/depth so GUI analysis does not stop at a small finite
    // Modern-B* proof sweep.
    bool infinite = false;
    bool ponder = false;
    int ponderhit_movetime_ms = 0;
};

struct Result {
    Move best_move = MOVE_NONE;
    Value score = VALUE_ZERO;

    // UCI-visible completed depth. For Modern-B* this is not the requested
    // proof budget, but the completed proof-depth estimate derived from root
    // refinement progress. This keeps Arena/CuteChess from seeing fake depth 32
    // after a short movetime search.
    int depth = 0;

    // Maximum tactical/verification depth touched by the root proof driver.
    int seldepth = 0;

    // Modern-B* diagnostics; harmless for ClassicAB.
    int requested_depth = 0;
    int proof_iterations = 0;
    int proof_effort = 0;
    int frontier_expansions = 0;
    int tree_nodes = 0;
    bool hard_proof = false;
    bool practical_proof = false;

    std::uint64_t nodes = 0;
    std::uint64_t time_ms = 0;
    std::string pv;
    std::string search_info;
    std::string debug_info;
    std::string score_source = "unknown";
    Value lower = VALUE_NONE;
    Value upper = VALUE_NONE;
    int confidence = 0;
    bool degraded = false;
};

class Searcher {
public:
    using InfoCallback = std::function<void(std::string_view)>;

    explicit Searcher(std::size_t hash_mb = 16);

    void set_hash(std::size_t hash_mb);
    void clear_hash();
    void stop();
    void reset_stop();
    void ponderhit();
    // UCI owns formatting/output synchronization. The search only emits
    // complete already-formatted info lines through this callback.
    void set_info_callback(InfoCallback callback);
    void clear_info_callback();

    [[nodiscard]] Result think(Position& root, const Limits& limits);
    [[nodiscard]] int hashfull() const;

private:
    using Clock = std::chrono::steady_clock;

    scarlet::tt::TranspositionTable tt_;
    SearchControl control_{};
    Limits limits_{};
    Clock::time_point start_{};
    Clock::time_point ponderhit_start_{};
    bool ponderhit_active_ = false;
    std::uint64_t nodes_ = 0;
    Move root_best_ = MOVE_NONE;
    std::mutex info_mutex_;
    InfoCallback info_callback_{};

    // Classic AB is deliberately retained only as HCE tactical sanitation.
    [[nodiscard]] bool should_stop();
    void emit_info(std::string_view line);
    [[nodiscard]] Value qsearch(Position& pos, Value alpha, Value beta, int ply,
                                bool& completed, bool hce_only,
                                NNUE::Accumulator* accumulator);
    [[nodiscard]] Value negamax(Position& pos, int depth, Value alpha, Value beta, int ply,
                                bool& completed, bool hce_only,
                                NNUE::Accumulator* accumulator);

    // Modern B* root/proof driver.
    [[nodiscard]] Result think_modern_bstar(Position& root, const Limits& limits);
    [[nodiscard]] Result think_classic_ab(Position& root, const Limits& limits);
    [[nodiscard]] std::string extract_pv(Position& root, int max_depth);
};

} // namespace Scarlet::Search

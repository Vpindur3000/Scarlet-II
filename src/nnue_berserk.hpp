#pragma once

#include "types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace Scarlet::NNUE {

inline constexpr int BERSERK_ACCUMULATOR_SIZE = 1024;

// Accumulators are deliberately search-path local. A shared Modern-B* DAG
// node may have several parents, while an accumulator belongs to one concrete
// make/unmake path into that node.
struct alignas(64) Accumulator {
    std::array<std::array<std::int16_t, BERSERK_ACCUMULATOR_SIZE>, 2> values{};
    bool valid = false;
};

struct BerserkStatus {
    bool enabled = true;
    bool loaded = false;
    std::string path;
    std::string message;
    std::uint64_t evals = 0;
    std::uint64_t accumulator_rebuilds = 0;
    std::uint64_t accumulator_updates = 0;
    std::uint64_t total_ns = 0;
};

// Scarlet-side port of Berserk 14's NNUE network format and accumulator math.
class BerserkNNUE final {
public:
    static BerserkNNUE& instance();

    BerserkNNUE(const BerserkNNUE&) = delete;
    BerserkNNUE& operator=(const BerserkNNUE&) = delete;

    bool load(std::string_view path);
    bool try_load_default();

    void set_enabled(bool enabled);
    [[nodiscard]] bool enabled() const;
    [[nodiscard]] bool loaded() const;
    [[nodiscard]] const BerserkStatus& status() const;

    // Score is Berserk-normalized UCI centipawns from side-to-move perspective.
    // Supplying a valid path-local accumulator avoids a full feature rebuild.
    [[nodiscard]] Value evaluate(const Position& pos, const Accumulator* accumulator = nullptr) const;

    void refresh(Accumulator& accumulator, const Position& pos) const;
    void apply_move(Accumulator& accumulator, const Position& after, Move move,
                    const StateInfo& state) const;
    [[nodiscard]] bool matches_full_rebuild(const Accumulator& accumulator,
                                            const Position& pos) const;

private:
    BerserkNNUE();

    [[nodiscard]] int propagate(const Position& pos, const Accumulator* accumulator) const;
    void copy_network_blob(const std::uint8_t* data, std::size_t size);

    BerserkStatus status_{};
};

} // namespace Scarlet::NNUE

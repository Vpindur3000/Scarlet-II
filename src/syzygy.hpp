#pragma once

#include "types.hpp"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace Scarlet::Syzygy {

// Five-valued Syzygy WDL from the side-to-move perspective. Cursed outcomes
// are deliberately not promoted to a win/loss score inside the search: DTZ at
// the root remains the authority for the current 50-move clock.
enum class Wdl : std::int8_t { Loss = -2, BlessedLoss = -1, Draw = 0, CursedWin = 1, Win = 2 };

struct RootProbe {
    Move best_move = MOVE_NONE;
    Wdl wdl = Wdl::Draw;
    unsigned dtz = 0;
};

struct Status {
    bool enabled = true;
    bool loaded = false;
    std::string path;
    int largest = 0;
    int probe_limit = 6;
    std::uint64_t wdl_probes = 0;
    std::uint64_t wdl_hits = 0;
    std::uint64_t root_probes = 0;
    std::uint64_t root_hits = 0;
    std::uint64_t skipped_castling = 0;
    std::uint64_t skipped_rule50 = 0;
    std::uint64_t failures = 0;
    std::string message;
};

class Backend final {
public:
    static Backend& instance();

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    ~Backend();

    void set_enabled(bool enabled);
    void set_probe_limit(int pieces);
    bool set_path(std::string_view path);

    [[nodiscard]] const Status& status() const { return status_; }
    // Thread-safe Pyrrhic WDL probe for use inside search. To avoid claiming
    // a 50-move-rule result from WDL alone, only zero-clock nodes are used.
    [[nodiscard]] std::optional<Wdl> probe_wdl(const Position& pos);
    // DTZ root probe honours the current 50-move clock and selects a legal
    // WDL-preserving move. Pyrrhic documents this operation as root-only.
    [[nodiscard]] std::optional<RootProbe> probe_root(const Position& pos);

private:
    Backend() = default;

    [[nodiscard]] bool eligible(const Position& pos) const;
    [[nodiscard]] static std::optional<Wdl> decode_wdl(unsigned raw);
    [[nodiscard]] static Move resolve_move(const Position& pos, unsigned raw);

    Status status_{};
    std::mutex config_mutex_;
    std::mutex root_mutex_;
};

} // namespace Scarlet::Syzygy

#pragma once

// Scarlet TT — single-file transposition table for copyleft chess engines.
//
// Design goals:
//   * Reckless-style 32-byte cluster: 3 x 8-byte payload entries + one u64 key/meta word.
//   * Stockfish-style semantics: generation aging, depth-preferred replacement, PV/exact priority,
//     mate-distance correction, TT move preservation, careful bound usage.
//   * Native interval/corridor payload for Modern B*: move/lower/upper/depth/flags.
//   * Optional sidecar static-eval cache without breaking the 32-byte main cluster layout.
//   * Racy TT friendly, but C++ data-race safe via lightweight relaxed atomic wrappers.
//
// Suggested project attribution when used in a GPL/AGPL engine README:
//   "Transposition table design inspired by public GPL/AGPL chess-engine techniques from
//    Stockfish and Reckless; implementation is original for this project."
//
// Minimum language: C++20.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <atomic>

#if defined(__linux__)
    #include <sys/mman.h>
    #include <unistd.h>
#endif

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #include <xmmintrin.h>
#endif

namespace scarlet::tt {

using Key   = std::uint64_t;
using Move  = std::uint16_t;   // 0 is treated as "no move" by default.
using Value = int;

inline constexpr Move MOVE_NONE = 0;

// Keep these values compatible with int16_t storage.
// Adapt them if your engine uses a different score scale, but keep |VALUE_MATE| < 32767.
inline constexpr Value VALUE_NONE            = 32767;
inline constexpr Value VALUE_INFINITE        = 32001;
inline constexpr Value VALUE_MATE            = 32000;
inline constexpr int   MAX_PLY               = 246;
inline constexpr Value VALUE_MATE_IN_MAX_PLY = VALUE_MATE - MAX_PLY;
inline constexpr Value VALUE_MATED_IN_MAX_PLY = -VALUE_MATE_IN_MAX_PLY;

[[nodiscard]] constexpr bool is_value_none(Value v) noexcept {
    return v == VALUE_NONE;
}

[[nodiscard]] constexpr bool is_valid_storable_value(Value v) noexcept {
    return v >= std::numeric_limits<std::int16_t>::min()
        && v <= std::numeric_limits<std::int16_t>::max();
}

// Convert a mate score from root-relative distance into TT-relative distance.
// This is the same semantic correction used by serious AB/PVS engines.
[[nodiscard]] constexpr Value value_to_tt(Value v, int ply) noexcept {
    if (v == VALUE_NONE)
        return v;
    if (v >= VALUE_MATE_IN_MAX_PLY)
        return v + ply;
    if (v <= VALUE_MATED_IN_MAX_PLY)
        return v - ply;
    return v;
}

// Convert a mate score from TT-relative distance back into root-relative distance.
[[nodiscard]] constexpr Value value_from_tt(Value v, int ply) noexcept {
    if (v == VALUE_NONE)
        return v;
    if (v >= VALUE_MATE_IN_MAX_PLY)
        return v - ply;
    if (v <= VALUE_MATED_IN_MAX_PLY)
        return v + ply;
    return v;
}

// Bound bits are intentionally AB-compatible, while the payload itself stores a full interval.
enum class Bound : std::uint8_t {
    None  = 0,
    Exact = 1,
    Lower = 2,
    Upper = 3,
};

[[nodiscard]] constexpr bool is_cutoff_bound(Bound b) noexcept {
    return b == Bound::Exact || b == Bound::Lower || b == Bound::Upper;
}

struct TTData {
    Move  move      = MOVE_NONE;
    Value lower     = VALUE_NONE;  // Native corridor lower bound, or lower AB bound.
    Value upper     = VALUE_NONE;  // Native corridor upper bound, or upper AB bound.
    Value static_eval = VALUE_NONE;
    int   depth     = -1;
    Bound bound     = Bound::None;
    bool  is_pv     = false;
    bool  has_move  = false;
    bool  has_classic_score = false;
    bool  has_corridor = false;
    bool  has_static_eval = false;
    bool  is_move_only = false;

    [[nodiscard]] bool occupied() const noexcept {
        return has_move || has_classic_score || has_corridor || has_static_eval || bound != Bound::None;
    }

    [[nodiscard]] bool has_usable_interval() const noexcept {
        return has_corridor && lower != VALUE_NONE && upper != VALUE_NONE && lower <= upper;
    }

    [[nodiscard]] bool has_exact_score() const noexcept {
        return has_classic_score && bound == Bound::Exact && lower != VALUE_NONE && lower == upper;
    }

    [[nodiscard]] Value exact_score_or_none() const noexcept {
        return has_exact_score() ? lower : VALUE_NONE;
    }
};

struct ProbeResult {
    bool hit = false;      // Verification key matched and entry is occupied.
    TTData data{};         // Always returns a TT move on hit, even if score interval is unusable.
};

namespace detail {

// Deterministic allocation fault injection for the TT strong-guarantee test.
// -1 disables it; 0 fails the next allocation; N fails after N successes.
inline std::atomic<int> allocation_failure_countdown{-1};

inline void set_allocation_failure_countdown(int value) noexcept {
    allocation_failure_countdown.store(value, std::memory_order_relaxed);
}

// C++ data-race safe relaxed access with the storage size/layout of the raw integer.
// std::atomic<T> may add ABI constraints and is not always pleasant inside packed entries;
// atomic_ref lets the stored object remain trivially copyable and memset-clearable when no search is running.
template <class T>
struct RelaxedAtomic {
    static_assert(std::is_integral_v<T>, "RelaxedAtomic is intended for integral TT fields");

    T raw{};

    [[nodiscard]] T load() const noexcept {
        return std::atomic_ref<const T>(raw).load(std::memory_order_relaxed);
    }

    void store(T v) noexcept {
        std::atomic_ref<T>(raw).store(v, std::memory_order_relaxed);
    }

    bool compare_exchange_weak(T& expected, T desired) noexcept {
        return std::atomic_ref<T>(raw).compare_exchange_weak(
            expected, desired, std::memory_order_relaxed, std::memory_order_relaxed);
    }
};

static_assert(std::is_trivially_copyable_v<RelaxedAtomic<std::uint16_t>>);
static_assert(sizeof(RelaxedAtomic<std::uint16_t>) == sizeof(std::uint16_t));
static_assert(sizeof(RelaxedAtomic<std::uint8_t>)  == sizeof(std::uint8_t));
static_assert(sizeof(RelaxedAtomic<std::uint64_t>) == sizeof(std::uint64_t));

inline constexpr std::uint8_t AGE_BITS  = 5;
inline constexpr std::uint8_t AGE_CYCLE = 1u << AGE_BITS;
inline constexpr std::uint8_t AGE_MASK  = AGE_CYCLE - 1u;

inline constexpr std::uint8_t BOUND_MASK = 0b0000'0011;
inline constexpr std::uint8_t PV_MASK    = 0b0000'0100;
inline constexpr std::uint8_t AGE_SHIFT  = 3;

[[nodiscard]] constexpr std::uint8_t make_flags(Bound b, bool pv, std::uint8_t age) noexcept {
    return (static_cast<std::uint8_t>(b) & BOUND_MASK)
         | (pv ? PV_MASK : 0)
         | ((age & AGE_MASK) << AGE_SHIFT);
}

[[nodiscard]] constexpr Bound flag_bound(std::uint8_t flags) noexcept {
    return static_cast<Bound>(flags & BOUND_MASK);
}

[[nodiscard]] constexpr bool flag_pv(std::uint8_t flags) noexcept {
    return (flags & PV_MASK) != 0;
}

[[nodiscard]] constexpr std::uint8_t flag_age(std::uint8_t flags) noexcept {
    return (flags >> AGE_SHIFT) & AGE_MASK;
}

[[nodiscard]] constexpr int relative_age(std::uint8_t entryAge, std::uint8_t currentAge) noexcept {
    return (AGE_CYCLE + currentAge - entryAge) & AGE_MASK;
}

[[nodiscard]] constexpr std::uint8_t depth_to_tt(int depth) noexcept {
    // 0 encodes qsearch-ish depth -1. Occupancy is flags != None, not depth != 0.
    const int packed = std::clamp(depth + 1, 0, 255);
    return static_cast<std::uint8_t>(packed);
}

[[nodiscard]] constexpr int depth_from_tt(std::uint8_t packed) noexcept {
    return static_cast<int>(packed) - 1;
}

[[nodiscard]] constexpr std::uint16_t verification_key(Key key) noexcept {
    return static_cast<std::uint16_t>(key);
}

[[nodiscard]] constexpr std::size_t mul_hi64(Key key, std::size_t n) noexcept {
#if defined(__SIZEOF_INT128__)
    return static_cast<std::size_t>((static_cast<unsigned __int128>(key) * n) >> 64);
#else
    // Portable fallback. Slightly slower, but only used on platforms without 128-bit integers.
    return n ? static_cast<std::size_t>(key % n) : 0;
#endif
}

struct EntryStorage {
    RelaxedAtomic<std::uint16_t> move16;
    RelaxedAtomic<std::int16_t>  lower16;
    RelaxedAtomic<std::int16_t>  upper16;
    RelaxedAtomic<std::uint8_t>  depth8;
    RelaxedAtomic<std::uint8_t>  flags8;
};

static_assert(sizeof(EntryStorage) == 8, "TT entry must stay exactly 8 bytes");
static_assert(alignof(EntryStorage) <= 2, "Unexpected entry alignment");

struct alignas(32) Cluster {
    EntryStorage entries[3];
    RelaxedAtomic<std::uint64_t> keysAndMeta; // low 48: 3 x key16; high 16: spare metadata bits.
};

static_assert(sizeof(Cluster) == 32, "TT cluster must stay exactly 32 bytes");
static_assert(alignof(Cluster) == 32, "TT cluster must be 32-byte aligned");

inline constexpr std::uint64_t KEY_LANE_MASKS[3] = {
    0x0000'0000'0000'FFFFull,
    0x0000'0000'FFFF'0000ull,
    0x0000'FFFF'0000'0000ull,
};
inline constexpr std::uint64_t META_BASE_SHIFT = 48;

[[nodiscard]] inline std::uint16_t key_at(std::uint64_t packed, int lane) noexcept {
    return static_cast<std::uint16_t>((packed >> (lane * 16)) & 0xFFFFu);
}

inline constexpr std::uint64_t meta_bit(int base, int lane) noexcept {
    return 1ull << (META_BASE_SHIFT + base + lane);
}

[[nodiscard]] inline bool eval_valid_at(std::uint64_t packed, int lane) noexcept {
    return (packed & meta_bit(0, lane)) != 0;
}

[[nodiscard]] inline bool move_only_at(std::uint64_t packed, int lane) noexcept {
    return (packed & meta_bit(3, lane)) != 0;
}

[[nodiscard]] inline bool classic_valid_at(std::uint64_t packed, int lane) noexcept {
    return (packed & meta_bit(6, lane)) != 0;
}

[[nodiscard]] inline bool corridor_valid_at(std::uint64_t packed, int lane) noexcept {
    return (packed & meta_bit(9, lane)) != 0;
}

[[nodiscard]] inline bool lane_occupied_at(std::uint64_t packed, int lane) noexcept {
    return eval_valid_at(packed, lane) || move_only_at(packed, lane)
        || classic_valid_at(packed, lane) || corridor_valid_at(packed, lane);
}

[[nodiscard]] inline int lookup_lane(std::uint64_t packed, std::uint16_t key16) noexcept {
    // Scalar/unrolled lookup is typically as fast as the bit trick for only three lanes,
    // and avoids matching the spare metadata lane by accident.
    return key_at(packed, 0) == key16 ? 0
         : key_at(packed, 1) == key16 ? 1
         : key_at(packed, 2) == key16 ? 2
         : -1;
}

inline void set_key_and_meta_bits(Cluster& c,
                                  int lane,
                                  std::uint16_t key16,
                                  bool hasEval,
                                  bool moveOnly,
                                  bool hasClassic,
                                  bool hasCorridor) noexcept {
    const std::uint64_t keyMask  = KEY_LANE_MASKS[lane];
    const std::uint64_t metaMask = meta_bit(0, lane) | meta_bit(3, lane) | meta_bit(6, lane) | meta_bit(9, lane);
    const std::uint64_t shiftedKey = std::uint64_t(key16) << (lane * 16);

    std::uint64_t old = c.keysAndMeta.load();
    for (;;) {
        std::uint64_t desired = (old & ~keyMask) | shiftedKey;
        desired &= ~metaMask;
        if (hasEval)     desired |= meta_bit(0, lane);
        if (moveOnly)    desired |= meta_bit(3, lane);
        if (hasClassic)  desired |= meta_bit(6, lane);
        if (hasCorridor) desired |= meta_bit(9, lane);
        if (c.keysAndMeta.compare_exchange_weak(old, desired))
            return;
    }
}

inline void set_eval_bit(Cluster& c, int lane, bool hasEval) noexcept {
    const std::uint64_t metaMask = meta_bit(0, lane);
    std::uint64_t old = c.keysAndMeta.load();
    for (;;) {
        std::uint64_t desired = hasEval ? (old | metaMask) : (old & ~metaMask);
        if (c.keysAndMeta.compare_exchange_weak(old, desired))
            return;
    }
}

struct SideEvalStorage {
    RelaxedAtomic<std::int16_t> eval16;
};

static_assert(sizeof(SideEvalStorage) == 2);

class MemoryBlock {
public:
    MemoryBlock() = default;
    MemoryBlock(const MemoryBlock&) = delete;
    MemoryBlock& operator=(const MemoryBlock&) = delete;

    MemoryBlock(MemoryBlock&& other) noexcept { move_from(other); }

    MemoryBlock& operator=(MemoryBlock&& other) noexcept {
        if (this != &other) {
            release();
            move_from(other);
        }
        return *this;
    }

    ~MemoryBlock() { release(); }

    [[nodiscard]] void* data() const noexcept { return ptr_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

    static MemoryBlock allocate(std::size_t bytes, std::size_t alignment, bool hugePageHint) {
        if (bytes == 0)
            return {};

        int countdown = allocation_failure_countdown.load(std::memory_order_relaxed);
        if (countdown >= 0) {
            countdown = allocation_failure_countdown.fetch_sub(1, std::memory_order_relaxed);
            if (countdown == 0) {
                allocation_failure_countdown.store(-1, std::memory_order_relaxed);
                throw std::bad_alloc();
            }
        }

        bytes = round_up(bytes, alignment);

#if defined(__linux__)
        void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED)
            throw std::bad_alloc();
        if (hugePageHint) {
            // Transparent huge pages when available. Failure is only a performance issue.
            (void)::madvise(p, bytes, MADV_HUGEPAGE);
        }
        return MemoryBlock(p, bytes, Kind::MMap);
#elif defined(_WIN32)
        (void)alignment;
        (void)hugePageHint;
        void* p = ::VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!p)
            throw std::bad_alloc();
        return MemoryBlock(p, bytes, Kind::VirtualAlloc);
#else
        (void)hugePageHint;
        void* p = ::operator new(bytes, std::align_val_t(alignment));
        return MemoryBlock(p, bytes, Kind::AlignedNew, alignment);
#endif
    }

private:
    enum class Kind { None, MMap, VirtualAlloc, AlignedNew };

    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
    std::size_t alignment_ = 64;
    Kind kind_ = Kind::None;

    MemoryBlock(void* p, std::size_t bytes, Kind k, std::size_t alignment = 64) noexcept
        : ptr_(p), bytes_(bytes), alignment_(alignment), kind_(k) {}

    static std::size_t round_up(std::size_t x, std::size_t a) noexcept {
        return (x + a - 1) / a * a;
    }

    void move_from(MemoryBlock& other) noexcept {
        ptr_ = other.ptr_;
        bytes_ = other.bytes_;
        alignment_ = other.alignment_;
        kind_ = other.kind_;
        other.ptr_ = nullptr;
        other.bytes_ = 0;
        other.kind_ = Kind::None;
    }

    void release() noexcept {
        if (!ptr_)
            return;

#if defined(__linux__)
        if (kind_ == Kind::MMap) {
            (void)::munmap(ptr_, bytes_);
        }
        else
#endif
#if defined(_WIN32)
        if (kind_ == Kind::VirtualAlloc) {
            (void)::VirtualFree(ptr_, 0, MEM_RELEASE);
        }
        else
#endif
        if (kind_ == Kind::AlignedNew) {
            ::operator delete(ptr_, std::align_val_t(alignment_));
        }

        ptr_ = nullptr;
        bytes_ = 0;
        kind_ = Kind::None;
    }
};

inline std::size_t default_thread_count() noexcept {
    const unsigned hc = std::thread::hardware_concurrency();
    return hc ? hc : 1;
}

} // namespace detail

enum class EvalCache : bool {
    Disabled = false,
    Sidecar  = true,
};

struct Config {
    std::size_t hash_mb = 16;
    EvalCache eval_cache = EvalCache::Sidecar;
    bool huge_pages = true;

    // Total UCI Hash accounting: when eval sidecar is enabled, cluster_count is reduced so that
    // main clusters + side eval cache stay roughly inside hash_mb.
    bool count_sidecar_in_hash = true;
};

class TranspositionTable {
public:
    TranspositionTable() { resize(Config{}); }
    explicit TranspositionTable(const Config& cfg) { resize(cfg); }

    TranspositionTable(const TranspositionTable&) = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;

    [[nodiscard]] std::size_t cluster_count() const noexcept { return clusterCount_; }
    [[nodiscard]] std::size_t hash_bytes() const noexcept { return clusters_.bytes() + sideEval_.bytes(); }
    [[nodiscard]] std::uint8_t generation() const noexcept { return generation_.load() & detail::AGE_MASK; }
    [[nodiscard]] bool has_eval_cache() const noexcept { return evalCache_ == EvalCache::Sidecar; }

    void resize(const Config& cfg) {
        const std::size_t requestedBytes = std::max<std::size_t>(1, cfg.hash_mb) * 1024ull * 1024ull;
        const EvalCache newEvalCache = cfg.eval_cache;
        const std::size_t sidePerCluster = newEvalCache == EvalCache::Sidecar ? 3 * sizeof(detail::SideEvalStorage) : 0;
        const std::size_t accountingPerCluster = sizeof(detail::Cluster)
                                           + (cfg.count_sidecar_in_hash ? sidePerCluster : 0);
        const std::size_t newClusterCount = std::max<std::size_t>(1, requestedBytes / accountingPerCluster);

        // Strong exception guarantee: build and initialize a complete new
        // storage object before publishing any size or pointer.
        auto newClusters = detail::MemoryBlock::allocate(
            newClusterCount * sizeof(detail::Cluster), alignof(detail::Cluster), cfg.huge_pages);
        detail::MemoryBlock newSideEval;
        if (newEvalCache == EvalCache::Sidecar)
            newSideEval = detail::MemoryBlock::allocate(
                newClusterCount * 3 * sizeof(detail::SideEvalStorage),
                alignof(detail::SideEvalStorage), cfg.huge_pages);
        std::memset(newClusters.data(), 0, newClusters.bytes());
        if (newSideEval.data()) std::memset(newSideEval.data(), 0, newSideEval.bytes());

        clusters_ = std::move(newClusters);
        sideEval_ = std::move(newSideEval);
        clusterCount_ = newClusterCount;
        evalCache_ = newEvalCache;
        generation_.store(0);
    }

    // Starts a new root search. 5-bit age wraps naturally.
    void new_search() noexcept {
        generation_.store(static_cast<std::uint8_t>((generation() + 1) & detail::AGE_MASK));
    }

    // Standard clear using temporary std::threads. For best NUMA placement in a serious engine,
    // call clear_with_thread_pool() from already pinned worker threads.
    void clear(std::size_t threads = detail::default_thread_count(), std::span<const int> workerToNuma = {}) {
        generation_.store(0);
        threads = std::max<std::size_t>(1, threads);

        std::vector<std::thread> pool;
        pool.reserve(threads);

        auto order = clear_order(threads, workerToNuma);
        for (std::size_t task = 0; task < threads; ++task) {
            pool.emplace_back([this, task, threads, order] {
                (void)order; // Kept to mirror clear_with_thread_pool ordering semantics.
                clear_range(task, threads);
            });
        }
        for (auto& t : pool)
            t.join();
    }

    // NUMA-aware first-touch variant for an engine thread pool.
    // run_on_thread(threadIndex, callable) should schedule callable on that already bound worker.
    // wait_on_thread(threadIndex) should block until that worker has completed its callable.
    template <class RunOnThread, class WaitOnThread>
    void clear_with_thread_pool(std::size_t threadCount,
                                RunOnThread&& run_on_thread,
                                WaitOnThread&& wait_on_thread,
                                std::span<const int> workerToNuma = {}) {
        generation_.store(0);
        threadCount = std::max<std::size_t>(1, threadCount);
        auto order = clear_order(threadCount, workerToNuma);

        for (std::size_t task = 0; task < threadCount; ++task) {
            const std::size_t worker = order[task];
            run_on_thread(worker, [this, task, threadCount] { clear_range(task, threadCount); });
        }

        for (std::size_t task = 0; task < threadCount; ++task)
            wait_on_thread(order[task]);
    }

    [[nodiscard]] ProbeResult probe(Key key, int ply) const noexcept {
        ProbeResult out{};
        if (clusterCount_ == 0)
            return out;

        const std::uint16_t key16 = detail::verification_key(key);
        detail::Cluster& c = cluster_for(key);
        const std::uint64_t packedKeys = c.keysAndMeta.load();
        const int lane = detail::lookup_lane(packedKeys, key16);

        if (lane < 0)
            return out;

        const detail::EntryStorage& e = c.entries[lane];
        const std::uint8_t flags = e.flags8.load();
        const Bound b = detail::flag_bound(flags);
        const bool metaMoveOnly = detail::move_only_at(packedKeys, lane);
        const bool metaClassic  = detail::classic_valid_at(packedKeys, lane);
        const bool metaCorridor = detail::corridor_valid_at(packedKeys, lane);
        const bool metaEval     = detail::eval_valid_at(packedKeys, lane);
        if (b == Bound::None && !metaMoveOnly && !metaClassic && !metaCorridor && !metaEval)
            return out;

        out.hit = true;
        out.data.move  = e.move16.load();
        out.data.has_move = out.data.move != MOVE_NONE;
        out.data.lower = value_from_tt(static_cast<Value>(e.lower16.load()), ply);
        out.data.upper = value_from_tt(static_cast<Value>(e.upper16.load()), ply);
        out.data.depth = detail::depth_from_tt(e.depth8.load());
        out.data.bound = b;
        out.data.is_pv = detail::flag_pv(flags);
        out.data.is_move_only = metaMoveOnly;
        out.data.has_classic_score = metaClassic;
        out.data.has_corridor = metaCorridor;

        if (evalCache_ == EvalCache::Sidecar && metaEval) {
            out.data.static_eval = side_eval_at(index_of(key), lane).eval16.load();
            out.data.has_static_eval = out.data.static_eval != VALUE_NONE;
        }

        // Mixed racy reads or rare key16 collisions can yield broken intervals.
        // Keep the TT move, but prevent score cutoffs through helper predicates.
        return out;
    }

    // Native Modern B* / corridor write. Both lower and upper are cached in the 8-byte payload.
    void save_corridor(Key key,
                       Value lower,
                       Value upper,
                       int depth,
                       Move move,
                       int ply,
                       bool pv,
                       Value staticEval = VALUE_NONE,
                       bool force = false) noexcept {
        Bound b = Bound::None;
        if (lower != VALUE_NONE && upper != VALUE_NONE && lower == upper)
            b = Bound::Exact;
        else if (lower != VALUE_NONE && upper != VALUE_NONE)
            b = Bound::Lower; // full interval stored; Lower means AB fail-high use is potentially available.
        else if (lower != VALUE_NONE)
            b = Bound::Lower;
        else if (upper != VALUE_NONE)
            b = Bound::Upper;
        else
            return;

        write_entry(key, lower, upper, depth, move, ply, pv, b, staticEval, force, false, true);
    }

    // Classic AB/PVS write. Payload still stores lower/upper; for classic use they are equal to value.
    // Static eval goes to the optional sidecar cache.
    void save_classic(Key key,
                      Value value,
                      Bound bound,
                      int depth,
                      Move move,
                      Value staticEval,
                      int ply,
                      bool pv,
                      bool force = false) noexcept {
        if (!is_cutoff_bound(bound) || value == VALUE_NONE)
            return;

        Value lower = VALUE_NONE;
        Value upper = VALUE_NONE;
        if (bound == Bound::Exact) {
            lower = value;
            upper = value;
        } else if (bound == Bound::Lower) {
            lower = value;
        } else if (bound == Bound::Upper) {
            upper = value;
        }

        write_entry(key, lower, upper, depth, move, ply, pv, bound, staticEval, force, true, false);
    }

    // Save only a move, useful when the score is known to be unreliable but move ordering should benefit.
    void save_move(Key key, Move move, bool pv = false) noexcept {
        if (move == MOVE_NONE || clusterCount_ == 0)
            return;

        const std::uint16_t key16 = detail::verification_key(key);
        detail::Cluster& c = cluster_for(key);
        std::uint64_t packedKeys = c.keysAndMeta.load();
        int lane = detail::lookup_lane(packedKeys, key16);
        if (lane < 0)
            lane = replacement_lane(c);

        detail::EntryStorage& e = c.entries[lane];
        e.move16.store(move);
        e.lower16.store(static_cast<std::int16_t>(VALUE_NONE));
        e.upper16.store(static_cast<std::int16_t>(VALUE_NONE));
        e.depth8.store(detail::depth_to_tt(-1));
        e.flags8.store(detail::make_flags(Bound::None, pv, generation()));
        detail::set_key_and_meta_bits(c, lane, key16, false, true, false, false);
    }

    // UCI-style hashfull in permille. maxAge=0 counts only current generation.
    [[nodiscard]] int hashfull(int maxAge = 0) const noexcept {
        if (clusterCount_ == 0)
            return 0;

        const std::size_t sample = std::min<std::size_t>(clusterCount_, 1000);
        int count = 0;
        const std::uint8_t age = generation();
        for (std::size_t i = 0; i < sample; ++i) {
            const detail::Cluster& c = clusters()[i];
            for (const auto& e : c.entries) {
                const std::uint8_t flags = e.flags8.load();
                const std::uint64_t packedKeys = c.keysAndMeta.load();
                if ((detail::flag_bound(flags) != Bound::None || detail::lane_occupied_at(packedKeys, int(&e - c.entries)))
                    && detail::relative_age(detail::flag_age(flags), age) <= maxAge) {
                    ++count;
                }
            }
        }
        return static_cast<int>((1000ull * count) / (sample * 3));
    }

    void prefetch(Key key) const noexcept {
        if (clusterCount_ == 0)
            return;
        const void* ptr = static_cast<const void*>(&cluster_for(key));
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
        _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(ptr, 0, 3);
#else
        (void)ptr;
#endif
    }

    // AB/PVS cutoff helper. It deliberately leaves the TT move usable even when score cannot cut.
    [[nodiscard]] static bool can_cutoff_classic(const TTData& d,
                                                 int requiredDepth,
                                                 Value alpha,
                                                 Value beta,
                                                 Value& cutoffValue) noexcept {
        if (!d.has_classic_score || d.depth < requiredDepth)
            return false;

        if (d.has_exact_score()) {
            cutoffValue = d.lower;
            return true;
        }

        if (d.bound == Bound::Lower && d.lower != VALUE_NONE && d.lower >= beta) {
            cutoffValue = d.lower;
            return true;
        }

        if (d.bound == Bound::Upper && d.upper != VALUE_NONE && d.upper <= alpha) {
            cutoffValue = d.upper;
            return true;
        }

        return false;
    }

    // Corridor proof helpers for Modern B*.
    [[nodiscard]] static bool proves_fail_high(const TTData& d, int requiredDepth, Value beta) noexcept {
        return (d.has_corridor || d.has_classic_score) && d.depth >= requiredDepth && d.lower != VALUE_NONE && d.lower >= beta;
    }

    [[nodiscard]] static bool proves_fail_low(const TTData& d, int requiredDepth, Value alpha) noexcept {
        return (d.has_corridor || d.has_classic_score) && d.depth >= requiredDepth && d.upper != VALUE_NONE && d.upper <= alpha;
    }

private:
    detail::MemoryBlock clusters_;
    detail::MemoryBlock sideEval_;
    std::size_t clusterCount_ = 0;
    EvalCache evalCache_ = EvalCache::Sidecar;
    detail::RelaxedAtomic<std::uint8_t> generation_{};

    [[nodiscard]] detail::Cluster* clusters() noexcept {
        return static_cast<detail::Cluster*>(clusters_.data());
    }

    [[nodiscard]] const detail::Cluster* clusters() const noexcept {
        return static_cast<const detail::Cluster*>(clusters_.data());
    }

    [[nodiscard]] detail::SideEvalStorage* side_evals() noexcept {
        return static_cast<detail::SideEvalStorage*>(sideEval_.data());
    }

    [[nodiscard]] const detail::SideEvalStorage* side_evals() const noexcept {
        return static_cast<const detail::SideEvalStorage*>(sideEval_.data());
    }

    [[nodiscard]] std::size_t index_of(Key key) const noexcept {
        return detail::mul_hi64(key, clusterCount_);
    }

    [[nodiscard]] detail::Cluster& cluster_for(Key key) const noexcept {
        return const_cast<detail::Cluster&>(clusters()[index_of(key)]);
    }

    [[nodiscard]] detail::SideEvalStorage& side_eval_at(std::size_t clusterIndex, int lane) const noexcept {
        return const_cast<detail::SideEvalStorage&>(side_evals()[clusterIndex * 3 + lane]);
    }

    [[nodiscard]] std::vector<std::size_t> clear_order(std::size_t threads, std::span<const int> workerToNuma) const {
        std::vector<std::size_t> order(threads);
        for (std::size_t i = 0; i < threads; ++i)
            order[i] = i;

        if (workerToNuma.size() == threads) {
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return workerToNuma[a] < workerToNuma[b];
            });
        }
        return order;
    }

    void clear_range(std::size_t task, std::size_t tasks) noexcept {
        const std::size_t start = clusterCount_ * task / tasks;
        const std::size_t end   = clusterCount_ * (task + 1) / tasks;
        if (end > start)
            std::memset(static_cast<void*>(clusters() + start), 0, (end - start) * sizeof(detail::Cluster));

        if (evalCache_ == EvalCache::Sidecar) {
            const std::size_t evalStart = (clusterCount_ * 3) * task / tasks;
            const std::size_t evalEnd   = (clusterCount_ * 3) * (task + 1) / tasks;
            // No need to fill VALUE_NONE: validity is carried in high bits of keysAndMeta.
            if (evalEnd > evalStart)
                std::memset(static_cast<void*>(side_evals() + evalStart), 0, (evalEnd - evalStart) * sizeof(detail::SideEvalStorage));
        }
    }

    [[nodiscard]] int replacement_lane(const detail::Cluster& c) const noexcept {
        const std::uint8_t currentAge = generation();
        int bestLane = 0;
        int bestQuality = std::numeric_limits<int>::max();

        for (int lane = 0; lane < 3; ++lane) {
            const detail::EntryStorage& e = c.entries[lane];
            const std::uint8_t flags = e.flags8.load();
            const Bound b = detail::flag_bound(flags);
            const std::uint64_t packedKeys = c.keysAndMeta.load();
            if (b == Bound::None && !detail::lane_occupied_at(packedKeys, lane))
                return lane;

            const int age = detail::relative_age(detail::flag_age(flags), currentAge);
            const int depth = detail::depth_from_tt(e.depth8.load());
            const int exactBonus = b == Bound::Exact ? 6 : 0;
            const int pvBonus = detail::flag_pv(flags) ? 2 : 0;
            const int moveOnlyPenalty = detail::move_only_at(packedKeys, lane) ? 10 : 0;
            const int quality = depth - 8 * age + exactBonus + pvBonus - moveOnlyPenalty;

            if (quality < bestQuality) {
                bestQuality = quality;
                bestLane = lane;
            }
        }
        return bestLane;
    }

    [[nodiscard]] bool should_replace_same_key(const detail::EntryStorage& e,
                                               Bound newBound,
                                               int newDepth,
                                               bool newPv,
                                               bool force) const noexcept {
        if (force || newBound == Bound::Exact)
            return true;

        const std::uint8_t oldFlags = e.flags8.load();
        const Bound oldBound = detail::flag_bound(oldFlags);
        if (oldBound == Bound::None)
            return true;

        const int age = detail::relative_age(detail::flag_age(oldFlags), generation());
        if (age != 0)
            return true;

        const int oldDepth = detail::depth_from_tt(e.depth8.load());
        const int oldScore = oldDepth + (oldBound == Bound::Exact ? 6 : 0) + (detail::flag_pv(oldFlags) ? 2 : 0);
        const int newScore = newDepth + (newBound == Bound::Exact ? 6 : 0) + (newPv ? 2 : 0);

        // Stockfish-like "not much worse" threshold: shallow searches may refresh a useful move,
        // but should not crush a significantly better same-key entry.
        return newScore + 4 > oldScore;
    }

    void write_entry(Key key,
                     Value lower,
                     Value upper,
                     int depth,
                     Move move,
                     int ply,
                     bool pv,
                     Bound bound,
                     Value staticEval,
                     bool force,
                     bool hasClassicScore,
                     bool hasCorridor) noexcept {
        if (clusterCount_ == 0)
            return;

        const Value ttLowerGuard = value_to_tt(lower, ply);
        const Value ttUpperGuard = value_to_tt(upper, ply);
        if (!is_valid_storable_value(ttLowerGuard)
            || !is_valid_storable_value(ttUpperGuard)
            || !is_valid_storable_value(staticEval))
            return;

        const std::uint16_t key16 = detail::verification_key(key);
        const std::size_t clusterIndex = index_of(key);
        detail::Cluster& c = clusters()[clusterIndex];

        const std::uint64_t packedKeys = c.keysAndMeta.load();
        int lane = detail::lookup_lane(packedKeys, key16);
        const bool sameKey = lane >= 0;
        if (!sameKey)
            lane = replacement_lane(c);

        detail::EntryStorage& e = c.entries[lane];

        // TT move preservation: if caller has no new move for the same key, keep the old move.
        if (move != MOVE_NONE || !sameKey)
            e.move16.store(move);

        if (sameKey && !should_replace_same_key(e, bound, depth, pv, force)) {
            // Even when score payload is not replaced, refresh side eval if supplied.
            if (evalCache_ == EvalCache::Sidecar && staticEval != VALUE_NONE) {
                side_eval_at(clusterIndex, lane).eval16.store(static_cast<std::int16_t>(staticEval));
                detail::set_eval_bit(c, lane, true);
            }
            return;
        }

        const Value ttLower = ttLowerGuard;
        const Value ttUpper = ttUpperGuard;

        e.lower16.store(static_cast<std::int16_t>(ttLower));
        e.upper16.store(static_cast<std::int16_t>(ttUpper));
        e.depth8.store(detail::depth_to_tt(depth));

        const bool hasEval = evalCache_ == EvalCache::Sidecar && staticEval != VALUE_NONE;
        if (hasEval)
            side_eval_at(clusterIndex, lane).eval16.store(static_cast<std::int16_t>(staticEval));

        // Store flags before publishing key. On same-key updates the key is already visible, but reads are allowed
        // to be racy; helper predicates defend against inconsistent interval cutoffs.
        e.flags8.store(detail::make_flags(bound, pv, generation()));
        detail::set_key_and_meta_bits(c, lane, key16, hasEval, false, hasClassicScore, hasCorridor);
    }
};

} // namespace scarlet::tt

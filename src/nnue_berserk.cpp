#include "nnue_berserk.hpp"
#include "runtime_paths.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <tuple>
#include <vector>
#include <chrono>
#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Scarlet::NNUE {
namespace {

constexpr int N_KING_BUCKETS = 16;
constexpr int N_FEATURES = N_KING_BUCKETS * 12 * 64;
constexpr int N_HIDDEN = BERSERK_ACCUMULATOR_SIZE;
constexpr int N_L1 = 2 * N_HIDDEN;
constexpr int N_L2 = 16;
constexpr int N_L3 = 32;
constexpr int SPARSE_CHUNK_SIZE = 4;
constexpr int QUANT1_BITS = 5;
constexpr int QUANT2_BITS = 12;
constexpr double BERSERK_UCI_NORMALIZATION = 1.58;
constexpr std::size_t NETWORK_SIZE =
    sizeof(std::int16_t) * N_FEATURES * N_HIDDEN +
    sizeof(std::int16_t) * N_HIDDEN +
    sizeof(std::int8_t)  * N_L1 * N_L2 +
    sizeof(std::int32_t) * N_L2 +
    sizeof(std::int16_t) * N_L2 * N_L3 +
    sizeof(std::int32_t) * N_L3 +
    sizeof(std::int16_t) * N_L3 +
    sizeof(std::int32_t);

constexpr std::array<std::uint16_t, 64> KING_BUCKETS = {
    15, 15, 14, 14, 14, 14, 15, 15,
    15, 15, 14, 14, 14, 14, 15, 15,
    13, 13, 12, 12, 12, 12, 13, 13,
    13, 13, 12, 12, 12, 12, 13, 13,
    11, 10,  9,  8,  8,  9, 10, 11,
    11, 10,  9,  8,  8,  9, 10, 11,
     7,  6,  5,  4,  4,  5,  6,  7,
     3,  2,  1,  0,  0,  1,  2,  3
};

alignas(64) std::array<std::int16_t, N_FEATURES * N_HIDDEN> INPUT_WEIGHTS{};
alignas(64) std::array<std::int16_t, N_HIDDEN> INPUT_BIASES{};
alignas(64) std::array<std::int8_t,  N_L1 * N_L2> L1_WEIGHTS{};
alignas(64) std::array<std::int32_t, N_L2> L1_BIASES{};
alignas(64) std::array<std::int16_t, N_L2 * N_L3> L2_WEIGHTS{};
alignas(64) std::array<std::int32_t, N_L3> L2_BIASES{};
alignas(64) std::array<std::int16_t, N_L3> OUTPUT_WEIGHTS{};
alignas(64) std::int32_t OUTPUT_BIAS = 0;

std::mutex g_mutex;

int berserk_sq(int scarletSq) {
    return scarletSq ^ 56; // Scarlet A1=0, Berserk A8=0.
}

int berserk_piece(Piece pc) {
    const int color = color_of(pc) == WHITE ? 0 : 1;
    const int pt = int(type_of(pc)) - 1; // Berserk: PAWN=0..KING=5.
    return 2 * pt + color;              // WHITE_PAWN=0, BLACK_PAWN=1, ...
}

int piece_type(int berserkPiece) {
    return berserkPiece >> 1;
}

int weight_idx_scrambled(int idx) {
    return ((idx / SPARSE_CHUNK_SIZE) % (N_L1 / SPARSE_CHUNK_SIZE)
            * N_L2 * SPARSE_CHUNK_SIZE)
         + (idx / N_L1 * SPARSE_CHUNK_SIZE)
         + (idx % SPARSE_CHUNK_SIZE);
}

int berserk_phase(const Position& pos) {
    // Berserk 14 scales its raw NNUE output by [1.0, 1.5] using a 0..64
    // material phase before exposing a UCI-centipawn score.
    return std::clamp(
        3 * (popcount(pos.pieces[WHITE][KNIGHT]) + popcount(pos.pieces[BLACK][KNIGHT])
           + popcount(pos.pieces[WHITE][BISHOP]) + popcount(pos.pieces[BLACK][BISHOP]))
      + 5 * (popcount(pos.pieces[WHITE][ROOK]) + popcount(pos.pieces[BLACK][ROOK]))
      + 10 * (popcount(pos.pieces[WHITE][QUEEN]) + popcount(pos.pieces[BLACK][QUEEN])),
        0, 64);
}

int feature_idx(int berserkPiece, int berserkSq, int berserkKingSq, int view) {
    const int oP  = 6 * ((berserkPiece ^ view) & 0x1) + piece_type(berserkPiece);
    const int oK  = (7 * !(berserkKingSq & 4)) ^ (56 * view) ^ berserkKingSq;
    const int oSq = (7 * !(berserkKingSq & 4)) ^ (56 * view) ^ berserkSq;
    return KING_BUCKETS[oK] * 12 * 64 + oP * 64 + oSq;
}

void accumulator_add_feature(std::int16_t* acc, int feature) {
    const std::int16_t* w = INPUT_WEIGHTS.data() + std::size_t(feature) * N_HIDDEN;
#if defined(__AVX2__)
    for (int i = 0; i < N_HIDDEN; i += 16) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(w + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_add_epi16(a, b));
    }
#else
    for (int i = 0; i < N_HIDDEN; ++i) acc[i] = std::int16_t(acc[i] + w[i]);
#endif
}

void accumulator_remove_feature(std::int16_t* acc, int feature) {
    const std::int16_t* w = INPUT_WEIGHTS.data() + std::size_t(feature) * N_HIDDEN;
#if defined(__AVX2__)
    for (int i = 0; i < N_HIDDEN; i += 16) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(w + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_sub_epi16(a, b));
    }
#else
    for (int i = 0; i < N_HIDDEN; ++i) acc[i] = std::int16_t(acc[i] - w[i]);
#endif
}

void build_accumulator(const Position& pos, int view, std::int16_t* acc) {
    std::memcpy(acc, INPUT_BIASES.data(), sizeof(std::int16_t) * N_HIDDEN);

    const int kingSq = berserk_sq(pos.kingSq[view == 0 ? WHITE : BLACK]);
    for (int s = 0; s < 64; ++s) {
        const Piece pc = pos.board[s];
        if (pc == EMPTY) continue;
        const int f = feature_idx(berserk_piece(pc), berserk_sq(s), kingSq, view);
        accumulator_add_feature(acc, f);
    }
}

void adjust_piece_feature(std::int16_t* acc, Piece pc, int scarletSq, int kingBerserkSq,
                          int view, bool add) {
    if (pc == EMPTY) return;
    const int feature = feature_idx(berserk_piece(pc), berserk_sq(scarletSq), kingBerserkSq, view);
    if (add) accumulator_add_feature(acc, feature);
    else accumulator_remove_feature(acc, feature);
}

void rebuild_view(Accumulator& accumulator, const Position& pos, int view) {
    build_accumulator(pos, view, accumulator.values[view].data());
}

void input_crelu8(std::int8_t* out, const std::int16_t* white, const std::int16_t* black, int stm) {
    const std::int16_t* views[2] = {stm == 0 ? white : black, stm == 0 ? black : white};
    for (int v = 0; v < 2; ++v) {
        for (int i = 0; i < N_HIDDEN; ++i) {
            int x = views[v][i] >> QUANT1_BITS;
            x = std::clamp(x, 0, 127);
            out[v * N_HIDDEN + i] = static_cast<std::int8_t>(x);
        }
    }
}

void l1_affine(std::int32_t* dest, const std::int8_t* src) {
    // Scalar sparse-chunk implementation of Berserk's L1. The accumulator build is
    // AVX2-hot; keeping this stage alias-safe is more important than a fragile vector
    // store in the beta branch.
    std::copy(L1_BIASES.begin(), L1_BIASES.end(), dest);
    for (int chunk = 0; chunk < N_L1 / SPARSE_CHUNK_SIZE; ++chunk) {
        const auto* in = reinterpret_cast<const std::uint8_t*>(src + chunk * SPARSE_CHUNK_SIZE);
        if ((in[0] | in[1] | in[2] | in[3]) == 0) continue;
        const std::int8_t* w = L1_WEIGHTS.data() + chunk * N_L2 * SPARSE_CHUNK_SIZE;
        for (int o = 0; o < N_L2; ++o) {
            int sum = 0;
            for (int k = 0; k < SPARSE_CHUNK_SIZE; ++k)
                sum += int(in[k]) * int(w[o * SPARSE_CHUNK_SIZE + k]);
            dest[o] += sum;
        }
    }
    for (int i = 0; i < N_L2; ++i) dest[i] >>= QUANT1_BITS;
}

void relu16(std::int16_t* dest, const std::int32_t* src, int n) {
    for (int i = 0; i < n; ++i)
        dest[i] = static_cast<std::int16_t>(std::clamp(src[i], 0, 32767));
}

void l2_affine(std::int32_t* dest, const std::int16_t* src) {
    for (int o = 0; o < N_L3; ++o) {
        int sum = L2_BIASES[o];
        for (int i = 0; i < N_L2; ++i)
            sum += int(src[i]) * int(L2_WEIGHTS[o * N_L2 + i]);
        dest[o] = sum >> QUANT1_BITS;
    }
}

int l3_transform(const std::int16_t* src) {
    int sum = OUTPUT_BIAS;
    for (int i = 0; i < N_L3; ++i)
        sum += int(src[i]) * int(OUTPUT_WEIGHTS[i]);
    return sum;
}

std::vector<std::filesystem::path> default_network_paths() {
    const char* env = std::getenv("SCARLET_BERSERK_NNUE");
    std::vector<std::filesystem::path> paths;
    if (env && *env) paths.emplace_back(RuntimePaths::explicit_path(env));
    paths.emplace_back(RuntimePaths::default_asset("assets/networks/nnue.nn"));
    paths.emplace_back(RuntimePaths::default_asset("../assets/networks/nnue.nn"));
    return paths;
}

} // namespace

BerserkNNUE& BerserkNNUE::instance() {
    static BerserkNNUE nnue;
    return nnue;
}

BerserkNNUE::BerserkNNUE() = default;

bool BerserkNNUE::load(std::string_view pathView) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const std::filesystem::path path = RuntimePaths::explicit_path(pathView);
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        status_.loaded = false;
        status_.path = path.string();
        status_.message = "unable to open Berserk NNUE file";
        return false;
    }
    const std::streamsize size = in.tellg();
    if (size != static_cast<std::streamsize>(NETWORK_SIZE)) {
        status_.loaded = false;
        status_.path = path.string();
        status_.message = "bad Berserk NNUE size: expected " + std::to_string(NETWORK_SIZE) +
                          ", got " + std::to_string(size);
        return false;
    }
    in.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> blob(static_cast<std::size_t>(size));
    if (!in.read(reinterpret_cast<char*>(blob.data()), size)) {
        status_.loaded = false;
        status_.path = path.string();
        status_.message = "failed to read Berserk NNUE file";
        return false;
    }
    copy_network_blob(blob.data(), blob.size());
    status_.loaded = true;
    status_.path = path.string();
    status_.message = "Berserk 14 NNUE loaded (reference L1 layout, cp-normalized)";
    status_.evals = 0;
    status_.accumulator_rebuilds = 0;
    status_.accumulator_updates = 0;
    status_.total_ns = 0;
    return true;
}

bool BerserkNNUE::try_load_default() {
    for (const auto& p : default_network_paths()) {
        if (std::filesystem::exists(p) && load(p.string())) return true;
    }
    status_.message = "Berserk NNUE not loaded; using HCE fallback";
    return false;
}

void BerserkNNUE::copy_network_blob(const std::uint8_t* data, std::size_t size) {
    if (size != NETWORK_SIZE) return;
    std::size_t off = 0;
    auto take = [&](auto& dst) {
        using ArrayT = std::remove_reference_t<decltype(dst)>;
        constexpr std::size_t bytes = sizeof(typename ArrayT::value_type) * std::tuple_size<ArrayT>::value;
        std::memcpy(dst.data(), data + off, bytes);
        off += bytes;
    };
    take(INPUT_WEIGHTS);
    take(INPUT_BIASES);
    // The network file stores L1 output-major. Berserk's sparse 4-byte chunk
    // kernel consumes a different layout. The old beta copied the bytes
    // directly and therefore evaluated a different, effectively corrupted
    // network while still reporting the backend as loaded.
    std::array<std::int8_t, N_L1 * N_L2> l1Raw{};
    take(l1Raw);
    for (int idx = 0; idx < N_L1 * N_L2; ++idx)
        L1_WEIGHTS[weight_idx_scrambled(idx)] = l1Raw[idx];
    take(L1_BIASES);
    take(L2_WEIGHTS);
    take(L2_BIASES);
    take(OUTPUT_WEIGHTS);
    std::memcpy(&OUTPUT_BIAS, data + off, sizeof(OUTPUT_BIAS));
}

void BerserkNNUE::set_enabled(bool enabled) {
    status_.enabled = enabled;
}

bool BerserkNNUE::enabled() const { return status_.enabled; }
bool BerserkNNUE::loaded() const { return status_.loaded; }
const BerserkStatus& BerserkNNUE::status() const { return status_; }

void BerserkNNUE::refresh(Accumulator& accumulator, const Position& pos) const {
    if (!status_.enabled || !status_.loaded) {
        accumulator.valid = false;
        return;
    }
    rebuild_view(accumulator, pos, 0);
    rebuild_view(accumulator, pos, 1);
    accumulator.valid = true;
    const_cast<BerserkStatus&>(status_).accumulator_rebuilds += 2;
}

void BerserkNNUE::apply_move(Accumulator& accumulator, const Position& after, Move move,
                             const StateInfo& state) const {
    if (!status_.enabled || !status_.loaded) {
        accumulator.valid = false;
        return;
    }
    if (!accumulator.valid) {
        refresh(accumulator, after);
        return;
    }

    const Piece moved = state.moved;
    const Color us = color_of(moved);
    const int from = from_sq(move);
    const int to = to_sq(move);
    const uint32_t flags = move_flags(move);
    const bool kingMoved = type_of(moved) == KING;

    int rookFrom = -1;
    int rookTo = -1;
    if (flags & MF_KING_CASTLE) {
        rookFrom = us == WHITE ? SQ_H1 : SQ_H8;
        rookTo = us == WHITE ? SQ_F1 : SQ_F8;
    } else if (flags & MF_QUEEN_CASTLE) {
        rookFrom = us == WHITE ? SQ_A1 : SQ_A8;
        rookTo = us == WHITE ? SQ_D1 : SQ_D8;
    }

    auto& status = const_cast<BerserkStatus&>(status_);
    for (int view = 0; view < 2; ++view) {
        const Color viewColor = view == 0 ? WHITE : BLACK;
        if (kingMoved && viewColor == us) {
            rebuild_view(accumulator, after, view);
            ++status.accumulator_rebuilds;
            continue;
        }

        const int kingSq = berserk_sq(after.kingSq[viewColor]);
        std::int16_t* values = accumulator.values[view].data();
        adjust_piece_feature(values, moved, from, kingSq, view, false);
        adjust_piece_feature(values, state.captured, state.capturedSq, kingSq, view, false);
        adjust_piece_feature(values, after.board[to], to, kingSq, view, true);
        if (rookFrom >= 0) {
            adjust_piece_feature(values, make_piece(us, ROOK), rookFrom, kingSq, view, false);
            adjust_piece_feature(values, after.board[rookTo], rookTo, kingSq, view, true);
        }
    }
    status.accumulator_updates += 2;
}

bool BerserkNNUE::matches_full_rebuild(const Accumulator& accumulator, const Position& pos) const {
    if (!status_.enabled || !status_.loaded) return !accumulator.valid;
    if (!accumulator.valid) return false;
    Accumulator rebuilt;
    rebuild_view(rebuilt, pos, 0);
    rebuild_view(rebuilt, pos, 1);
    rebuilt.valid = true;
    return rebuilt.values == accumulator.values;
}

Value BerserkNNUE::evaluate(const Position& pos, const Accumulator* accumulator) const {
    if (!status_.enabled || !status_.loaded) return VALUE_NONE;
    auto& st = const_cast<BerserkStatus&>(status_);
    const auto t0 = std::chrono::steady_clock::now();
    const int v = propagate(pos, accumulator && accumulator->valid ? accumulator : nullptr);
    const auto t1 = std::chrono::steady_clock::now();
    ++st.evals;
    st.total_ns += std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    return v;
}

int BerserkNNUE::propagate(const Position& pos, const Accumulator* accumulator) const {
    alignas(64) std::array<std::int16_t, N_HIDDEN> white{};
    alignas(64) std::array<std::int16_t, N_HIDDEN> black{};
    alignas(64) std::array<std::int8_t, N_L1> x0{};
    alignas(64) std::array<std::int32_t, N_L3> dest{};
    alignas(64) std::array<std::int16_t, N_L3> act{};

    const std::int16_t* whiteValues = nullptr;
    const std::int16_t* blackValues = nullptr;
    if (accumulator) {
        whiteValues = accumulator->values[0].data();
        blackValues = accumulator->values[1].data();
    } else {
        build_accumulator(pos, 0, white.data());
        build_accumulator(pos, 1, black.data());
        const_cast<BerserkStatus&>(status_).accumulator_rebuilds += 2;
        whiteValues = white.data();
        blackValues = black.data();
    }

    input_crelu8(x0.data(), whiteValues, blackValues, pos.side == WHITE ? 0 : 1);
    l1_affine(dest.data(), x0.data());
    relu16(act.data(), dest.data(), N_L2);
    l2_affine(dest.data(), act.data());
    relu16(act.data(), dest.data(), N_L3);
    const int raw = l3_transform(act.data()) >> QUANT2_BITS;
    const int scaled = (128 + berserk_phase(pos)) * raw / 128;
    const int cp = int(std::lround(double(scaled) / BERSERK_UCI_NORMALIZATION));
    return std::clamp(cp, -VALUE_MATE_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1);
}

} // namespace Scarlet::NNUE

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__BMI2__) && (defined(__x86_64__) || defined(_M_X64))
  #include <immintrin.h>
#endif

namespace Scarlet {

using Bitboard = uint64_t;
using Move     = uint32_t;
using TTMove   = uint16_t;
using Key      = uint64_t;

constexpr Move MOVE_NONE = 0;
constexpr TTMove TT_MOVE_NONE = 0;

constexpr Bitboard AllSquares = ~Bitboard(0);
constexpr Bitboard FileA = 0x0101010101010101ULL;
constexpr Bitboard FileB = FileA << 1;
constexpr Bitboard FileG = FileA << 6;
constexpr Bitboard FileH = FileA << 7;
constexpr Bitboard Rank1 = 0x00000000000000FFULL;
constexpr Bitboard Rank2 = 0x000000000000FF00ULL;
constexpr Bitboard Rank3 = 0x0000000000FF0000ULL;
constexpr Bitboard Rank4 = 0x00000000FF000000ULL;
constexpr Bitboard Rank5 = 0x000000FF00000000ULL;
constexpr Bitboard Rank6 = 0x0000FF0000000000ULL;
constexpr Bitboard Rank7 = 0x00FF000000000000ULL;
constexpr Bitboard Rank8 = 0xFF00000000000000ULL;

inline int pop_lsb(Bitboard& b) {
    assert(b);
#if defined(__GNUC__) || defined(__clang__)
    int s = __builtin_ctzll(b);
#else
    unsigned long idx;
    _BitScanForward64(&idx, b);
    int s = int(idx);
#endif
    b &= b - 1;
    return s;
}

inline int lsb(Bitboard b) {
    assert(b);
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(b);
#else
    unsigned long idx;
    _BitScanForward64(&idx, b);
    return int(idx);
#endif
}

inline int popcount(Bitboard b) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(b);
#else
    return int(__popcnt64(b));
#endif
}

constexpr int file_of(int s) { return s & 7; }
constexpr int rank_of(int s) { return s >> 3; }
constexpr Bitboard sq_bb(int s) { return Bitboard(1) << s; }

constexpr int SQ_A1 = 0, SQ_B1 = 1, SQ_C1 = 2, SQ_D1 = 3, SQ_E1 = 4, SQ_F1 = 5, SQ_G1 = 6, SQ_H1 = 7;
constexpr int SQ_A8 = 56, SQ_B8 = 57, SQ_C8 = 58, SQ_D8 = 59, SQ_E8 = 60, SQ_F8 = 61, SQ_G8 = 62, SQ_H8 = 63;

enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };
constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : int {
    NO_PIECE_TYPE = 0,
    PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6,
    PIECE_TYPE_NB = 7
};

enum Piece : int {
    EMPTY = 0,
    W_PAWN = 1, W_KNIGHT = 2, W_BISHOP = 3, W_ROOK = 4, W_QUEEN = 5, W_KING = 6,
    B_PAWN = 9, B_KNIGHT = 10, B_BISHOP = 11, B_ROOK = 12, B_QUEEN = 13, B_KING = 14
};

constexpr Piece make_piece(Color c, PieceType pt) { return Piece((c == WHITE ? 0 : 8) + pt); }
constexpr Color color_of(Piece p) { assert(p != EMPTY); return p >= B_PAWN ? BLACK : WHITE; }
constexpr PieceType type_of(Piece p) { return PieceType(int(p) & 7); }

inline std::string square_name(int s) {
    std::string out;
    out += char('a' + file_of(s));
    out += char('1' + rank_of(s));
    return out;
}

// Move encoding: from[0..5], to[6..11], promo piece type[12..15], flags[16..23].
enum MoveFlag : uint32_t {
    MF_NONE       = 0,
    MF_CAPTURE    = 1u << 16,
    MF_DOUBLE     = 1u << 17,
    MF_KING_CASTLE= 1u << 18,
    MF_QUEEN_CASTLE=1u << 19,
    MF_EN_PASSANT = 1u << 20,
    MF_PROMOTION  = 1u << 21
};

constexpr Move make_move(int from, int to, uint32_t flags = MF_NONE, PieceType promo = NO_PIECE_TYPE) {
    return Move(from | (to << 6) | (int(promo) << 12) | flags);
}
constexpr int from_sq(Move m) { return int(m & 0x3F); }
constexpr int to_sq(Move m) { return int((m >> 6) & 0x3F); }
constexpr PieceType promotion_type(Move m) { return PieceType((m >> 12) & 0xF); }
constexpr uint32_t move_flags(Move m) { return m & 0x00FF0000u; }
constexpr bool is_capture(Move m) { return move_flags(m) & MF_CAPTURE; }
constexpr bool is_promotion(Move m) { return move_flags(m) & MF_PROMOTION; }
constexpr bool is_en_passant(Move m) { return move_flags(m) & MF_EN_PASSANT; }
constexpr bool is_castle(Move m) { return move_flags(m) & (MF_KING_CASTLE | MF_QUEEN_CASTLE); }

// TTMove contract:
//   Move    is the executable/search move: from/to/promo core + generated flags.
//   TTMove  stores only the stable 16-bit UCI identity: from, to, promotion type.
// A TT move must never be executed directly after probe. Resolve it against the
// current pseudo-legal move list to recover capture/castling/en-passant flags.
constexpr uint32_t MOVE_CORE_MASK = 0xFFFFu;
static_assert(((uint32_t(MF_CAPTURE) | uint32_t(MF_DOUBLE) | uint32_t(MF_KING_CASTLE) |
                uint32_t(MF_QUEEN_CASTLE) | uint32_t(MF_EN_PASSANT) | uint32_t(MF_PROMOTION))
               & MOVE_CORE_MASK) == 0,
              "Move flags must stay outside the 16-bit TTMove core");
constexpr TTMove move_to_tt(Move m) { return TTMove(m & MOVE_CORE_MASK); }
constexpr bool same_move_core(Move m, TTMove tt) { return move_to_tt(m) == tt; }

inline std::string move_to_uci(Move m) {
    std::string s = square_name(from_sq(m)) + square_name(to_sq(m));
    if (is_promotion(m)) {
        switch (promotion_type(m)) {
            case KNIGHT: s += 'n'; break;
            case BISHOP: s += 'b'; break;
            case ROOK:   s += 'r'; break;
            case QUEEN:  s += 'q'; break;
            default: break;
        }
    }
    return s;
}

struct MoveList {
    std::array<Move, 256> moves{};
    int size = 0;

    void push(Move m) {
        assert(size < int(moves.size()));
        moves[size++] = m;
    }
    Move* begin() { return moves.data(); }
    Move* end() { return moves.data() + size; }
    const Move* begin() const { return moves.data(); }
    const Move* end() const { return moves.data() + size; }
};

namespace Attacks {

inline std::array<Bitboard, 64> King{};
inline std::array<Bitboard, 64> Knight{};
inline std::array<std::array<Bitboard, 64>, 2> Pawn{};

struct SliderTable {
    std::array<Bitboard, 64> masks{};
    std::array<int, 64> offsets{};
    std::vector<Bitboard> attacks;
};

inline SliderTable BishopTable;
inline SliderTable RookTable;

inline uint64_t pext_soft(uint64_t x, uint64_t mask) {
    uint64_t res = 0;
    for (uint64_t bb = 1; mask; bb <<= 1) {
        uint64_t l = mask & -mask;
        if (x & l) res |= bb;
        mask -= l;
    }
    return res;
}

inline uint64_t pext_u64(uint64_t x, uint64_t mask) {
#if defined(__BMI2__) && (defined(__x86_64__) || defined(_M_X64))
    return _pext_u64(x, mask);
#else
    return pext_soft(x, mask);
#endif
}

inline Bitboard ray_attacks_from(int s, Bitboard occ, const int deltas[], int n) {
    Bitboard attacks = 0;
    int sf = file_of(s), sr = rank_of(s);

    for (int i = 0; i < n; ++i) {
        int d = deltas[i];
        int df = (d == 1 || d == 9 || d == -7) ? 1 : (d == -1 || d == 7 || d == -9) ? -1 : 0;
        int dr = (d == 8 || d == 9 || d == 7) ? 1 : (d == -8 || d == -9 || d == -7) ? -1 : 0;
        int f = sf + df, r = sr + dr;
        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
            int to = r * 8 + f;
            attacks |= sq_bb(to);
            if (occ & sq_bb(to)) break;
            f += df; r += dr;
        }
    }
    return attacks;
}

inline Bitboard bishop_attacks_slow(int s, Bitboard occ) {
    static constexpr int D[4] = { 9, 7, -7, -9 };
    return ray_attacks_from(s, occ, D, 4);
}

inline Bitboard rook_attacks_slow(int s, Bitboard occ) {
    static constexpr int D[4] = { 8, -8, 1, -1 };
    return ray_attacks_from(s, occ, D, 4);
}

inline Bitboard bishop_mask(int s) {
    Bitboard mask = 0;
    int sf = file_of(s), sr = rank_of(s);
    const int df[4] = { 1, -1, 1, -1 };
    const int dr[4] = { 1, 1, -1, -1 };
    for (int i = 0; i < 4; ++i) {
        int f = sf + df[i], r = sr + dr[i];
        while (f > 0 && f < 7 && r > 0 && r < 7) {
            mask |= sq_bb(r * 8 + f);
            f += df[i]; r += dr[i];
        }
    }
    return mask;
}

inline Bitboard rook_mask(int s) {
    Bitboard mask = 0;
    int sf = file_of(s), sr = rank_of(s);
    for (int r = sr + 1; r <= 6; ++r) mask |= sq_bb(r * 8 + sf);
    for (int r = sr - 1; r >= 1; --r) mask |= sq_bb(r * 8 + sf);
    for (int f = sf + 1; f <= 6; ++f) mask |= sq_bb(sr * 8 + f);
    for (int f = sf - 1; f >= 1; --f) mask |= sq_bb(sr * 8 + f);
    return mask;
}

inline void init_slider_table(SliderTable& t, bool bishop) {
    int total = 0;
    for (int s = 0; s < 64; ++s) {
        Bitboard mask = bishop ? bishop_mask(s) : rook_mask(s);
        t.masks[s] = mask;
        t.offsets[s] = total;
        total += 1 << popcount(mask);
    }
    t.attacks.assign(total, 0);

    for (int s = 0; s < 64; ++s) {
        Bitboard mask = t.masks[s];
        Bitboard subset = 0;
        while (true) {
            uint64_t idx = pext_soft(subset, mask);
            t.attacks[t.offsets[s] + int(idx)] = bishop ? bishop_attacks_slow(s, subset)
                                                       : rook_attacks_slow(s, subset);
            subset = (subset - mask) & mask;
            if (!subset) break;
        }
    }
}

inline void init_leapers() {
    for (int s = 0; s < 64; ++s) {
        int f = file_of(s), r = rank_of(s);
        Bitboard k = 0, n = 0;

        for (int df = -1; df <= 1; ++df)
            for (int dr = -1; dr <= 1; ++dr)
                if (df || dr) {
                    int nf = f + df, nr = r + dr;
                    if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
                        k |= sq_bb(nr * 8 + nf);
                }
        King[s] = k;

        constexpr int KDF[8] = { 1, 2, 2, 1, -1, -2, -2, -1 };
        constexpr int KDR[8] = { 2, 1, -1, -2, -2, -1, 1, 2 };
        for (int i = 0; i < 8; ++i) {
            int nf = f + KDF[i], nr = r + KDR[i];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
                n |= sq_bb(nr * 8 + nf);
        }
        Knight[s] = n;

        Bitboard wp = 0, bp = 0;
        if (f > 0 && r < 7) wp |= sq_bb((r + 1) * 8 + (f - 1));
        if (f < 7 && r < 7) wp |= sq_bb((r + 1) * 8 + (f + 1));
        if (f > 0 && r > 0) bp |= sq_bb((r - 1) * 8 + (f - 1));
        if (f < 7 && r > 0) bp |= sq_bb((r - 1) * 8 + (f + 1));
        Pawn[WHITE][s] = wp;
        Pawn[BLACK][s] = bp;
    }
}

inline void init() {
    init_leapers();
    init_slider_table(BishopTable, true);
    init_slider_table(RookTable, false);
}

inline Bitboard bishop_attacks(int s, Bitboard occ) {
    const Bitboard mask = BishopTable.masks[s];
    const int idx = BishopTable.offsets[s] + int(pext_u64(occ, mask));
    return BishopTable.attacks[idx];
}

inline Bitboard rook_attacks(int s, Bitboard occ) {
    const Bitboard mask = RookTable.masks[s];
    const int idx = RookTable.offsets[s] + int(pext_u64(occ, mask));
    return RookTable.attacks[idx];
}

inline Bitboard queen_attacks(int s, Bitboard occ) {
    return bishop_attacks(s, occ) | rook_attacks(s, occ);
}

} // namespace Attacks


namespace Zobrist {

inline std::array<std::array<Key, 64>, 16> pieceSquare{};
inline Key sideKey = 0;
inline std::array<Key, 16> castlingKey{};
inline std::array<Key, 8> epFileKey{};
inline bool initialized = false;

inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline void init() {
    uint64_t seed = 0x5C4A'521E'7A9B'D13FULL;
    for (auto& bySquare : pieceSquare)
        for (Key& k : bySquare)
            k = splitmix64(seed);
    sideKey = splitmix64(seed);
    for (Key& k : castlingKey) k = splitmix64(seed);
    for (Key& k : epFileKey)   k = splitmix64(seed);
    initialized = true;
}

} // namespace Zobrist

constexpr int LEELA_HISTORY_FRAMES = 8;

struct LeelaHistoryFrame {
    // Compact absolute-colour board snapshot. The raw Leela encoder reorients
    // every frame to the current side to move when it builds the 112 planes.
    std::array<Bitboard, 12> pieces{};
    bool repeated = false;
};

struct StateInfo {
    int castlingRights = 0;
    int epSquare = -1;
    int halfmoveClock = 0;
    Piece captured = EMPTY;
    int capturedSq = -1;
    Piece moved = EMPTY;
    Key key = 0;
    Key pawnKey = 0;
    std::size_t historySize = 0;
    int leelaHistoryCount = 0;
    bool leelaHistoryDropped = false;
    LeelaHistoryFrame leelaDroppedFrame{};
};

enum CastlingRight : int { WK = 1, WQ = 2, BK = 4, BQ = 8 };

struct Position {
    std::array<std::array<Bitboard, PIECE_TYPE_NB>, COLOR_NB> pieces{};
    std::array<Bitboard, COLOR_NB> occ{};
    Bitboard occAll = 0;
    std::array<Piece, 64> board{};
    std::array<int, COLOR_NB> kingSq{ SQ_E1, SQ_E8 };
    Color side = WHITE;
    int castlingRights = 0;
    int epSquare = -1;
    int halfmoveClock = 0;
    int fullmoveNumber = 1;
    Key key = 0;
    Key pawnKey = 0;
    std::vector<Key> keyHistory{};
    std::array<LeelaHistoryFrame, LEELA_HISTORY_FRAMES> leelaHistory{};
    int leelaHistoryCount = 0;

    void clear() {
        pieces = {};
        occ = {};
        occAll = 0;
        board.fill(EMPTY);
        kingSq = { SQ_E1, SQ_E8 };
        side = WHITE;
        castlingRights = 0;
        epSquare = -1;
        halfmoveClock = 0;
        fullmoveNumber = 1;
        key = 0;
        pawnKey = 0;
        keyHistory.clear();
        leelaHistory = {};
        leelaHistoryCount = 0;
    }

    [[nodiscard]] LeelaHistoryFrame leela_history_frame(bool repeated) const {
        LeelaHistoryFrame frame;
        for (int c = 0; c < COLOR_NB; ++c)
            for (int pt = PAWN; pt <= KING; ++pt)
                frame.pieces[c * 6 + (pt - PAWN)] = pieces[c][pt];
        frame.repeated = repeated;
        return frame;
    }

    void append_leela_history(bool repeated, StateInfo& st) {
        st.leelaHistoryDropped = leelaHistoryCount == LEELA_HISTORY_FRAMES;
        if (st.leelaHistoryDropped) {
            st.leelaDroppedFrame = leelaHistory[0];
            for (int i = 1; i < LEELA_HISTORY_FRAMES; ++i)
                leelaHistory[i - 1] = leelaHistory[i];
            leelaHistory[LEELA_HISTORY_FRAMES - 1] = leela_history_frame(repeated);
        } else {
            leelaHistory[leelaHistoryCount++] = leela_history_frame(repeated);
        }
    }

    void restore_leela_history(const StateInfo& st) {
        if (st.leelaHistoryDropped) {
            for (int i = LEELA_HISTORY_FRAMES - 1; i > 0; --i)
                leelaHistory[i] = leelaHistory[i - 1];
            leelaHistory[0] = st.leelaDroppedFrame;
        }
        leelaHistoryCount = st.leelaHistoryCount;
    }

    void put_piece(Piece pc, int s) {
        assert(pc != EMPTY);
        Color c = color_of(pc);
        PieceType pt = type_of(pc);
        board[s] = pc;
        pieces[c][pt] |= sq_bb(s);
        occ[c] |= sq_bb(s);
        occAll |= sq_bb(s);
        key ^= Zobrist::pieceSquare[int(pc)][s];
        if (pt == PAWN) pawnKey ^= Zobrist::pieceSquare[int(pc)][s];
        if (pt == KING) kingSq[c] = s;
    }

    void remove_piece(int s) {
        Piece pc = board[s];
        assert(pc != EMPTY);
        Color c = color_of(pc);
        PieceType pt = type_of(pc);
        Bitboard b = sq_bb(s);
        board[s] = EMPTY;
        pieces[c][pt] &= ~b;
        occ[c] &= ~b;
        occAll &= ~b;
        key ^= Zobrist::pieceSquare[int(pc)][s];
        if (pt == PAWN) pawnKey ^= Zobrist::pieceSquare[int(pc)][s];
    }

    void move_piece(int from, int to) {
        Piece pc = board[from];
        assert(pc != EMPTY && board[to] == EMPTY);
        Color c = color_of(pc);
        PieceType pt = type_of(pc);
        Bitboard fb = sq_bb(from), tb = sq_bb(to);
        board[from] = EMPTY;
        board[to] = pc;
        pieces[c][pt] ^= fb | tb;
        occ[c] ^= fb | tb;
        occAll ^= fb | tb;
        key ^= Zobrist::pieceSquare[int(pc)][from] ^ Zobrist::pieceSquare[int(pc)][to];
        if (pt == PAWN) pawnKey ^= Zobrist::pieceSquare[int(pc)][from] ^ Zobrist::pieceSquare[int(pc)][to];
        if (pt == KING) kingSq[c] = to;
    }

    bool set_fen(const std::string& fen) {
        clear();

        auto fail = [this]() {
            clear();
            return false;
        };

        std::istringstream ss(fen);
        std::array<std::string, 6> fields{};
        std::string field;
        int fieldCount = 0;
        while (ss >> field) {
            if (fieldCount == int(fields.size())) return fail();
            fields[fieldCount++] = field;
        }
        if (fieldCount != 4 && fieldCount != 6) return fail();

        const std::string& placement = fields[0];
        const std::string& stm       = fields[1];
        const std::string& castling  = fields[2];
        const std::string& ep        = fields[3];

        if (stm == "w") side = WHITE;
        else if (stm == "b") side = BLACK;
        else return fail();

        auto parse_uint = [](const std::string& s, int& out) {
            if (s.empty()) return false;
            int value = 0;
            for (char ch : s) {
                if (ch < '0' || ch > '9') return false;
                value = value * 10 + (ch - '0');
                if (value > 1000000) return false;
            }
            out = value;
            return true;
        };

        halfmoveClock = 0;
        fullmoveNumber = 1;
        if (fieldCount == 6) {
            if (!parse_uint(fields[4], halfmoveClock)) return fail();
            if (!parse_uint(fields[5], fullmoveNumber) || fullmoveNumber <= 0) return fail();
        }

        std::array<std::string, 8> ranks{};
        std::string cur;
        int rankCount = 0;
        for (char ch : placement) {
            if (ch == '/') {
                if (rankCount >= 8 || cur.empty()) return fail();
                ranks[rankCount++] = cur;
                cur.clear();
            } else {
                cur += ch;
            }
        }
        if (rankCount >= 8 || cur.empty()) return fail();
        ranks[rankCount++] = cur;
        if (rankCount != 8) return fail();

        int whiteKings = 0;
        int blackKings = 0;

        for (int ri = 0; ri < 8; ++ri) {
            const int r = 7 - ri;
            int f = 0;
            bool previousWasDigit = false;

            for (char ch : ranks[ri]) {
                if (ch >= '1' && ch <= '8') {
                    if (previousWasDigit) return fail();
                    previousWasDigit = true;
                    f += ch - '0';
                    if (f > 8) return fail();
                    continue;
                }

                previousWasDigit = false;
                if (f >= 8) return fail();

                Piece pc = EMPTY;
                switch (ch) {
                    case 'P': pc = W_PAWN; break; case 'N': pc = W_KNIGHT; break;
                    case 'B': pc = W_BISHOP; break; case 'R': pc = W_ROOK; break;
                    case 'Q': pc = W_QUEEN; break; case 'K': pc = W_KING; break;
                    case 'p': pc = B_PAWN; break; case 'n': pc = B_KNIGHT; break;
                    case 'b': pc = B_BISHOP; break; case 'r': pc = B_ROOK; break;
                    case 'q': pc = B_QUEEN; break; case 'k': pc = B_KING; break;
                    default: return fail();
                }

                if (type_of(pc) == PAWN && (r == 0 || r == 7)) return fail();
                if (pc == W_KING) ++whiteKings;
                if (pc == B_KING) ++blackKings;
                put_piece(pc, r * 8 + f);
                ++f;
            }

            if (f != 8) return fail();
        }

        if (whiteKings != 1 || blackKings != 1) return fail();

        castlingRights = 0;
        if (castling != "-") {
            std::array<bool, 128> seen{};
            for (char ch : castling) {
                const unsigned char uch = static_cast<unsigned char>(ch);
                if (uch >= 128 || seen[uch]) return fail();
                seen[uch] = true;
                switch (ch) {
                    case 'K': castlingRights |= WK; break;
                    case 'Q': castlingRights |= WQ; break;
                    case 'k': castlingRights |= BK; break;
                    case 'q': castlingRights |= BQ; break;
                    default: return fail();
                }
            }
        }

        if ((castlingRights & WK) && !(board[SQ_E1] == W_KING && board[SQ_H1] == W_ROOK)) return fail();
        if ((castlingRights & WQ) && !(board[SQ_E1] == W_KING && board[SQ_A1] == W_ROOK)) return fail();
        if ((castlingRights & BK) && !(board[SQ_E8] == B_KING && board[SQ_H8] == B_ROOK)) return fail();
        if ((castlingRights & BQ) && !(board[SQ_E8] == B_KING && board[SQ_A8] == B_ROOK)) return fail();

        epSquare = -1;
        if (ep != "-") {
            if (ep.size() != 2) return fail();
            const int f = ep[0] - 'a';
            const int r = ep[1] - '1';
            if (f < 0 || f > 7 || r < 0 || r > 7) return fail();
            if ((side == WHITE && r != 5) || (side == BLACK && r != 2)) return fail();

            const int sq = r * 8 + f;
            if (board[sq] != EMPTY) return fail();

            const int pushedPawnSq = side == WHITE ? sq - 8 : sq + 8;
            const Piece pushedPawn = side == WHITE ? B_PAWN : W_PAWN;
            if (pushedPawnSq < 0 || pushedPawnSq >= 64 || board[pushedPawnSq] != pushedPawn)
                return fail();

            bool capturable = false;
            if (side == WHITE) {
                if (f > 0 && board[sq - 9] == W_PAWN) capturable = true;
                if (f < 7 && board[sq - 7] == W_PAWN) capturable = true;
            } else {
                if (f > 0 && board[sq + 7] == B_PAWN) capturable = true;
                if (f < 7 && board[sq + 9] == B_PAWN) capturable = true;
            }

            // FEN can name an ep target after any double pawn push. Internally Scarlet
            // keeps only capturable ep squares, so equivalent positions hash the same.
            epSquare = capturable ? sq : -1;
        }

        key ^= Zobrist::castlingKey[castlingRights & 15];
        if (epSquare != -1) key ^= Zobrist::epFileKey[file_of(epSquare)];
        if (side == BLACK) key ^= Zobrist::sideKey;

        keyHistory.push_back(key);
        leelaHistory[0] = leela_history_frame(false);
        leelaHistoryCount = 1;

        return true;
    }

    Bitboard pieces_bb(Color c, PieceType pt) const { return pieces[c][pt]; }

    bool square_attacked_by(int s, Color by, Bitboard occupied) const {
        using namespace Attacks;
        if (Pawn[~by][s] & pieces[by][PAWN]) return true;
        if (Knight[s] & pieces[by][KNIGHT]) return true;
        if (King[s] & pieces[by][KING]) return true;
        if (bishop_attacks(s, occupied) & (pieces[by][BISHOP] | pieces[by][QUEEN])) return true;
        if (rook_attacks(s, occupied) & (pieces[by][ROOK] | pieces[by][QUEEN])) return true;
        return false;
    }

    bool square_attacked_by(int s, Color by) const {
        return square_attacked_by(s, by, occAll);
    }

    bool in_check(Color c) const { return square_attacked_by(kingSq[c], ~c); }

    int repetition_count() const {
        if (keyHistory.size() < 3) return 0;
        const int reversible = std::min<int>(halfmoveClock, int(keyHistory.size()) - 1);
        const int current = int(keyHistory.size()) - 1;
        int count = 0;
        for (int back = 2; back <= reversible; back += 2)
            if (keyHistory[current - back] == key) ++count;
        return count;
    }

    bool is_repetition() const { return repetition_count() > 0; }

    void clear_castling_by_square(int s) {
        switch (s) {
            case SQ_E1: castlingRights &= ~(WK | WQ); break;
            case SQ_H1: castlingRights &= ~WK; break;
            case SQ_A1: castlingRights &= ~WQ; break;
            case SQ_E8: castlingRights &= ~(BK | BQ); break;
            case SQ_H8: castlingRights &= ~BK; break;
            case SQ_A8: castlingRights &= ~BQ; break;
            default: break;
        }
    }

    bool make_move(Move m, StateInfo& st) {
        const Color us = side;
        const Color them = ~us;
        const int from = from_sq(m);
        const int to = to_sq(m);
        const uint32_t flags = move_flags(m);
        const Piece moved = board[from];
        assert(moved != EMPTY && color_of(moved) == us);

        st.castlingRights = castlingRights;
        st.epSquare = epSquare;
        st.halfmoveClock = halfmoveClock;
        st.captured = EMPTY;
        st.capturedSq = -1;
        st.moved = moved;
        st.key = key;
        st.pawnKey = pawnKey;
        st.historySize = keyHistory.size();
        st.leelaHistoryCount = leelaHistoryCount;
        st.leelaHistoryDropped = false;

        key ^= Zobrist::castlingKey[castlingRights & 15];
        if (epSquare != -1) key ^= Zobrist::epFileKey[file_of(epSquare)];
        if (side == BLACK) key ^= Zobrist::sideKey;

        // Castling legality has one extra rule: the king may not cross an attacked square.
        // The transit square must be tested with the king removed from its origin; otherwise
        // x-ray attacks through e1/e8 can be missed. Final-square check is still done below
        // after the full make, because the rook also changes square.
        if (flags & (MF_KING_CASTLE | MF_QUEEN_CASTLE)) {
            const int mid = (flags & MF_KING_CASTLE)
                            ? (us == WHITE ? SQ_F1 : SQ_F8)
                            : (us == WHITE ? SQ_D1 : SQ_D8);
            const Bitboard occWithoutKing = occAll ^ sq_bb(from);
            if (square_attacked_by(from, them) || square_attacked_by(mid, them, occWithoutKing)) {
                key = st.key;
                pawnKey = st.pawnKey;
                return false;
            }
        }

        Piece captured = EMPTY;
        int capturedSq = to;
        if (flags & MF_EN_PASSANT) {
            capturedSq = us == WHITE ? to - 8 : to + 8;
            captured = board[capturedSq];
            assert(type_of(captured) == PAWN && color_of(captured) == them);
        } else {
            captured = board[to];
            assert(captured == EMPTY || color_of(captured) == them);
        }

        st.captured = captured;
        st.capturedSq = captured == EMPTY ? -1 : capturedSq;

        epSquare = -1;
        halfmoveClock++;
        if (type_of(moved) == PAWN || captured != EMPTY) halfmoveClock = 0;

        clear_castling_by_square(from);
        if (captured != EMPTY) clear_castling_by_square(capturedSq);

        remove_piece(from);
        if (captured != EMPTY) remove_piece(capturedSq);

        if (flags & MF_PROMOTION) {
            put_piece(make_piece(us, promotion_type(m)), to);
        } else {
            put_piece(moved, to);
        }

        if (flags & MF_KING_CASTLE) {
            if (us == WHITE) move_piece(SQ_H1, SQ_F1);
            else             move_piece(SQ_H8, SQ_F8);
        } else if (flags & MF_QUEEN_CASTLE) {
            if (us == WHITE) move_piece(SQ_A1, SQ_D1);
            else             move_piece(SQ_A8, SQ_D8);
        }

        if (flags & MF_DOUBLE) {
            const int ep = us == WHITE ? from + 8 : from - 8;
            const int f = file_of(ep);
            bool capturable = false;

            if (us == WHITE) {
                if (f > 0 && board[ep + 7] == B_PAWN) capturable = true;
                if (f < 7 && board[ep + 9] == B_PAWN) capturable = true;
            } else {
                if (f > 0 && board[ep - 9] == W_PAWN) capturable = true;
                if (f < 7 && board[ep - 7] == W_PAWN) capturable = true;
            }

            if (capturable) epSquare = ep;
        }

        side = them;
        if (us == BLACK) ++fullmoveNumber;

        key ^= Zobrist::castlingKey[castlingRights & 15];
        if (epSquare != -1) key ^= Zobrist::epFileKey[file_of(epSquare)];
        if (side == BLACK) key ^= Zobrist::sideKey;

        if (square_attacked_by(kingSq[us], them)) {
            undo_move(m, st);
            return false;
        }
        keyHistory.push_back(key);
        append_leela_history(is_repetition(), st);
        return true;
    }

    void undo_move(Move m, const StateInfo& st) {
        side = ~side;
        const Color us = side;
        const int from = from_sq(m);
        const int to = to_sq(m);
        const uint32_t flags = move_flags(m);

        if (us == BLACK) --fullmoveNumber;

        if (flags & MF_KING_CASTLE) {
            if (us == WHITE) move_piece(SQ_F1, SQ_H1);
            else             move_piece(SQ_F8, SQ_H8);
        } else if (flags & MF_QUEEN_CASTLE) {
            if (us == WHITE) move_piece(SQ_D1, SQ_A1);
            else             move_piece(SQ_D8, SQ_A8);
        }

        remove_piece(to);
        put_piece(st.moved, from);
        if (st.captured != EMPTY)
            put_piece(st.captured, st.capturedSq);

        castlingRights = st.castlingRights;
        epSquare = st.epSquare;
        halfmoveClock = st.halfmoveClock;
        key = st.key;
        pawnKey = st.pawnKey;
        keyHistory.resize(st.historySize);
        restore_leela_history(st);
    }
};

inline void add_promotions(MoveList& ml, int from, int to, uint32_t flags) {
    ml.push(make_move(from, to, flags | MF_PROMOTION, QUEEN));
    ml.push(make_move(from, to, flags | MF_PROMOTION, ROOK));
    ml.push(make_move(from, to, flags | MF_PROMOTION, BISHOP));
    ml.push(make_move(from, to, flags | MF_PROMOTION, KNIGHT));
}

inline void add_normal_or_promo(MoveList& ml, Color us, int from, int to, uint32_t flags) {
    const bool promo = us == WHITE ? rank_of(to) == 7 : rank_of(to) == 0;
    if (promo) add_promotions(ml, from, to, flags);
    else       ml.push(make_move(from, to, flags));
}

inline void generate_pawn_moves(const Position& p, MoveList& ml) {
    const Color us = p.side;
    const Color them = ~us;
    const Bitboard enemy = p.occ[them];
    const Bitboard empty = ~p.occAll;

    Bitboard pawns = p.pieces[us][PAWN];
    if (us == WHITE) {
        Bitboard one = (pawns << 8) & empty;
        Bitboard prom = one & Rank8;
        Bitboard quiet = one & ~Rank8;
        Bitboard b = quiet;
        while (b) {
            int to = pop_lsb(b);
            ml.push(make_move(to - 8, to));
        }
        b = prom;
        while (b) {
            int to = pop_lsb(b);
            add_promotions(ml, to - 8, to, MF_NONE);
        }

        Bitboard two = ((one & Rank3) << 8) & empty;
        b = two;
        while (b) {
            int to = pop_lsb(b);
            ml.push(make_move(to - 16, to, MF_DOUBLE));
        }

        Bitboard capW = ((pawns & ~FileA) << 7) & enemy;
        b = capW;
        while (b) {
            int to = pop_lsb(b);
            add_normal_or_promo(ml, WHITE, to - 7, to, MF_CAPTURE);
        }
        Bitboard capE = ((pawns & ~FileH) << 9) & enemy;
        b = capE;
        while (b) {
            int to = pop_lsb(b);
            add_normal_or_promo(ml, WHITE, to - 9, to, MF_CAPTURE);
        }
        if (p.epSquare != -1) {
            Bitboard ep = sq_bb(p.epSquare);
            Bitboard epW = ((pawns & ~FileA) << 7) & ep;
            Bitboard epE = ((pawns & ~FileH) << 9) & ep;
            if (epW) ml.push(make_move(p.epSquare - 7, p.epSquare, MF_CAPTURE | MF_EN_PASSANT));
            if (epE) ml.push(make_move(p.epSquare - 9, p.epSquare, MF_CAPTURE | MF_EN_PASSANT));
        }
    } else {
        Bitboard one = (pawns >> 8) & empty;
        Bitboard prom = one & Rank1;
        Bitboard quiet = one & ~Rank1;
        Bitboard b = quiet;
        while (b) {
            int to = pop_lsb(b);
            ml.push(make_move(to + 8, to));
        }
        b = prom;
        while (b) {
            int to = pop_lsb(b);
            add_promotions(ml, to + 8, to, MF_NONE);
        }

        Bitboard two = ((one & Rank6) >> 8) & empty;
        b = two;
        while (b) {
            int to = pop_lsb(b);
            ml.push(make_move(to + 16, to, MF_DOUBLE));
        }

        Bitboard capE = ((pawns & ~FileH) >> 7) & enemy;
        b = capE;
        while (b) {
            int to = pop_lsb(b);
            add_normal_or_promo(ml, BLACK, to + 7, to, MF_CAPTURE);
        }
        Bitboard capW = ((pawns & ~FileA) >> 9) & enemy;
        b = capW;
        while (b) {
            int to = pop_lsb(b);
            add_normal_or_promo(ml, BLACK, to + 9, to, MF_CAPTURE);
        }
        if (p.epSquare != -1) {
            Bitboard ep = sq_bb(p.epSquare);
            Bitboard epE = ((pawns & ~FileH) >> 7) & ep;
            Bitboard epW = ((pawns & ~FileA) >> 9) & ep;
            if (epE) ml.push(make_move(p.epSquare + 7, p.epSquare, MF_CAPTURE | MF_EN_PASSANT));
            if (epW) ml.push(make_move(p.epSquare + 9, p.epSquare, MF_CAPTURE | MF_EN_PASSANT));
        }
    }
}

inline void generate_knight_moves(const Position& p, MoveList& ml) {
    using namespace Attacks;
    const Color us = p.side;
    const Color them = ~us;
    const Bitboard own = p.occ[us];
    const Bitboard enemy = p.occ[them];

    Bitboard b = p.pieces[us][KNIGHT];
    while (b) {
        int from = pop_lsb(b);
        Bitboard targets = Knight[from] & ~own;
        while (targets) {
            int to = pop_lsb(targets);
            ml.push(make_move(from, to, (enemy & sq_bb(to)) ? MF_CAPTURE : MF_NONE));
        }
    }
}

inline void generate_slider_moves(const Position& p, MoveList& ml, PieceType pt) {
    using namespace Attacks;
    const Color us = p.side;
    const Color them = ~us;
    const Bitboard own = p.occ[us];
    const Bitboard enemy = p.occ[them];

    Bitboard b = p.pieces[us][pt];
    while (b) {
        int from = pop_lsb(b);
        Bitboard targets = 0;
        if (pt == BISHOP) targets = bishop_attacks(from, p.occAll);
        else if (pt == ROOK) targets = rook_attacks(from, p.occAll);
        else {
            assert(pt == QUEEN);
            targets = queen_attacks(from, p.occAll);
        }
        targets &= ~own;

        while (targets) {
            int to = pop_lsb(targets);
            ml.push(make_move(from, to, (enemy & sq_bb(to)) ? MF_CAPTURE : MF_NONE));
        }
    }
}

inline void generate_king_moves(const Position& p, MoveList& ml) {
    using namespace Attacks;
    const Color us = p.side;
    const Color them = ~us;
    const Bitboard own = p.occ[us];
    const Bitboard enemy = p.occ[them];

    const int ksq = p.kingSq[us];
    Bitboard targets = King[ksq] & ~own;
    while (targets) {
        int to = pop_lsb(targets);
        ml.push(make_move(ksq, to, (enemy & sq_bb(to)) ? MF_CAPTURE : MF_NONE));
    }
}

inline void generate_castling_moves(const Position& p, MoveList& ml) {
    const Color us = p.side;

    if (us == WHITE) {
        if ((p.castlingRights & WK) && p.board[SQ_E1] == W_KING && p.board[SQ_H1] == W_ROOK
            && !(p.occAll & (sq_bb(SQ_F1) | sq_bb(SQ_G1)))
            && !p.square_attacked_by(SQ_E1, BLACK)
            && !p.square_attacked_by(SQ_F1, BLACK, p.occAll ^ sq_bb(SQ_E1))
            && !p.square_attacked_by(SQ_G1, BLACK))
            ml.push(make_move(SQ_E1, SQ_G1, MF_KING_CASTLE));

        if ((p.castlingRights & WQ) && p.board[SQ_E1] == W_KING && p.board[SQ_A1] == W_ROOK
            && !(p.occAll & (sq_bb(SQ_D1) | sq_bb(SQ_C1) | sq_bb(SQ_B1)))
            && !p.square_attacked_by(SQ_E1, BLACK)
            && !p.square_attacked_by(SQ_D1, BLACK, p.occAll ^ sq_bb(SQ_E1))
            && !p.square_attacked_by(SQ_C1, BLACK))
            ml.push(make_move(SQ_E1, SQ_C1, MF_QUEEN_CASTLE));
    } else {
        if ((p.castlingRights & BK) && p.board[SQ_E8] == B_KING && p.board[SQ_H8] == B_ROOK
            && !(p.occAll & (sq_bb(SQ_F8) | sq_bb(SQ_G8)))
            && !p.square_attacked_by(SQ_E8, WHITE)
            && !p.square_attacked_by(SQ_F8, WHITE, p.occAll ^ sq_bb(SQ_E8))
            && !p.square_attacked_by(SQ_G8, WHITE))
            ml.push(make_move(SQ_E8, SQ_G8, MF_KING_CASTLE));

        if ((p.castlingRights & BQ) && p.board[SQ_E8] == B_KING && p.board[SQ_A8] == B_ROOK
            && !(p.occAll & (sq_bb(SQ_D8) | sq_bb(SQ_C8) | sq_bb(SQ_B8)))
            && !p.square_attacked_by(SQ_E8, WHITE)
            && !p.square_attacked_by(SQ_D8, WHITE, p.occAll ^ sq_bb(SQ_E8))
            && !p.square_attacked_by(SQ_C8, WHITE))
            ml.push(make_move(SQ_E8, SQ_C8, MF_QUEEN_CASTLE));
    }
}

inline void generate_pseudo_legal(const Position& p, MoveList& ml) {
    ml.size = 0;
    generate_pawn_moves(p, ml);
    generate_knight_moves(p, ml);
    generate_slider_moves(p, ml, BISHOP);
    generate_slider_moves(p, ml, ROOK);
    generate_slider_moves(p, ml, QUEEN);
    generate_king_moves(p, ml);
    generate_castling_moves(p, ml);
}

inline Move resolve_tt_move(const Position& p, TTMove ttMove) {
    if (ttMove == TT_MOVE_NONE) return MOVE_NONE;

    MoveList pseudo;
    generate_pseudo_legal(p, pseudo);
    for (Move m : pseudo)
        if (same_move_core(m, ttMove))
            return m;

    return MOVE_NONE;
}

inline void generate_legal(Position& p, MoveList& legal) {
    MoveList pseudo;
    generate_pseudo_legal(p, pseudo);
    legal.size = 0;
    for (Move m : pseudo) {
        StateInfo st;
        if (p.make_move(m, st)) {
            legal.push(m);
            p.undo_move(m, st);
        }
    }
}

inline uint64_t perft(Position& p, int depth) {
    if (depth == 0) return 1;

    MoveList pseudo;
    generate_pseudo_legal(p, pseudo);

    if (depth == 1) {
        uint64_t nodes = 0;
        for (Move m : pseudo) {
            StateInfo st;
            if (p.make_move(m, st)) {
                ++nodes;
                p.undo_move(m, st);
            }
        }
        return nodes;
    }

    uint64_t nodes = 0;
    for (Move m : pseudo) {
        StateInfo st;
        if (p.make_move(m, st)) {
            nodes += perft(p, depth - 1);
            p.undo_move(m, st);
        }
    }
    return nodes;
}

inline void perft_divide(Position& p, int depth) {
    MoveList pseudo;
    generate_pseudo_legal(p, pseudo);
    uint64_t total = 0;
    for (Move m : pseudo) {
        StateInfo st;
        if (p.make_move(m, st)) {
            uint64_t n = perft(p, depth - 1);
            p.undo_move(m, st);
            total += n;
            std::cout << move_to_uci(m) << ": " << n << "\n";
        }
    }
    std::cout << "total: " << total << "\n";
}


namespace Geometry {

inline std::array<std::array<Bitboard, 64>, 64> Between{};
inline std::array<std::array<Bitboard, 64>, 64> Line{};

inline bool same_line(int a, int b) {
    const int af = file_of(a), ar = rank_of(a);
    const int bf = file_of(b), br = rank_of(b);
    return af == bf || ar == br || std::abs(af - bf) == std::abs(ar - br);
}

inline void init() {
    for (int a = 0; a < 64; ++a) {
        for (int b = 0; b < 64; ++b) {
            Between[a][b] = 0;
            Line[a][b] = 0;
            if (a == b || !same_line(a, b)) continue;

            const int af = file_of(a), ar = rank_of(a);
            const int bf = file_of(b), br = rank_of(b);
            const int df = (bf > af) - (bf < af);
            const int dr = (br > ar) - (br < ar);

            int f = af + df, r = ar + dr;
            while (f >= 0 && f < 8 && r >= 0 && r < 8) {
                const int s = r * 8 + f;
                Line[a][b] |= sq_bb(s);
                if (s == b) break;
                Between[a][b] |= sq_bb(s);
                f += df; r += dr;
            }
            Line[a][b] |= sq_bb(a);
            f = af - df; r = ar - dr;
            while (f >= 0 && f < 8 && r >= 0 && r < 8) {
                Line[a][b] |= sq_bb(r * 8 + f);
                f -= df; r -= dr;
            }
            Between[a][b] &= ~(sq_bb(a) | sq_bb(b));
        }
    }
}

} // namespace Geometry

inline void init() {
    Attacks::init();
    Geometry::init();
    Zobrist::init();
}

inline Key recompute_key(const Position& p) {
    Key k = 0;
    for (int s = 0; s < 64; ++s) {
        Piece pc = p.board[s];
        if (pc != EMPTY) k ^= Zobrist::pieceSquare[int(pc)][s];
    }
    k ^= Zobrist::castlingKey[p.castlingRights & 15];
    if (p.epSquare != -1) k ^= Zobrist::epFileKey[file_of(p.epSquare)];
    if (p.side == BLACK) k ^= Zobrist::sideKey;
    return k;
}

inline Key recompute_pawn_key(const Position& p) {
    Key k = 0;
    for (int s = 0; s < 64; ++s) {
        Piece pc = p.board[s];
        if (pc != EMPTY && type_of(pc) == PAWN) k ^= Zobrist::pieceSquare[int(pc)][s];
    }
    return k;
}

inline bool same_core(Move a, Move b) {
    return (a & MOVE_CORE_MASK) == (b & MOVE_CORE_MASK);
}

inline bool legal(Position& p, Move m) {
    StateInfo st;
    if (!p.make_move(m, st)) return false;
    p.undo_move(m, st);
    return true;
}

inline bool pseudo_legal(const Position& p, Move m) {
    MoveList ml;
    generate_pseudo_legal(p, ml);
    for (Move x : ml)
        if (same_core(x, m)) return true;
    return false;
}

inline void generate_captures(const Position& p, MoveList& out) {
    MoveList all;
    generate_pseudo_legal(p, all);
    out.size = 0;
    for (Move m : all)
        if (is_capture(m) || is_promotion(m)) out.push(m);
}

inline void generate_quiets(const Position& p, MoveList& out) {
    MoveList all;
    generate_pseudo_legal(p, all);
    out.size = 0;
    for (Move m : all)
        if (!is_capture(m) && !is_promotion(m)) out.push(m);
}

inline void generate_evasions(Position& p, MoveList& out) {
    // Correct first implementation: legal moves are evasions when side is in check.
    // Later this can become a masked generator without changing the public API.
    generate_legal(p, out);
}

inline Bitboard attackers_to(const Position& p, int s, Bitboard occupied) {
    using namespace Attacks;
    Bitboard attackers = 0;
    attackers |= Pawn[BLACK][s] & p.pieces[WHITE][PAWN];
    attackers |= Pawn[WHITE][s] & p.pieces[BLACK][PAWN];
    attackers |= Knight[s] & (p.pieces[WHITE][KNIGHT] | p.pieces[BLACK][KNIGHT]);
    attackers |= King[s] & (p.pieces[WHITE][KING] | p.pieces[BLACK][KING]);
    attackers |= bishop_attacks(s, occupied) & (p.pieces[WHITE][BISHOP] | p.pieces[WHITE][QUEEN] |
                                                p.pieces[BLACK][BISHOP] | p.pieces[BLACK][QUEEN]);
    attackers |= rook_attacks(s, occupied) & (p.pieces[WHITE][ROOK] | p.pieces[WHITE][QUEEN] |
                                              p.pieces[BLACK][ROOK] | p.pieces[BLACK][QUEEN]);
    return attackers;
}

inline Bitboard attackers_to(const Position& p, int s) {
    return attackers_to(p, s, p.occAll);
}

inline Bitboard checkers(const Position& p) {
    return attackers_to(p, p.kingSq[p.side]) & p.occ[~p.side];
}

inline Bitboard pinned_pieces(const Position& p, Color c) {
    Bitboard pinned = 0;
    const int ksq = p.kingSq[c];
    const int kf = file_of(ksq), kr = rank_of(ksq);
    constexpr int DF[8] = { 1, -1, 0, 0, 1, 1, -1, -1 };
    constexpr int DR[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };

    for (int dir = 0; dir < 8; ++dir) {
        int f = kf + DF[dir], r = kr + DR[dir];
        int candidate = -1;
        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
            const int s = r * 8 + f;
            const Piece pc = p.board[s];
            if (pc != EMPTY) {
                if (color_of(pc) == c) {
                    if (candidate != -1) break;
                    candidate = s;
                } else {
                    const PieceType pt = type_of(pc);
                    const bool diagonal = DF[dir] != 0 && DR[dir] != 0;
                    const bool slider = diagonal ? (pt == BISHOP || pt == QUEEN)
                                                 : (pt == ROOK || pt == QUEEN);
                    if (candidate != -1 && slider) pinned |= sq_bb(candidate);
                    break;
                }
            }
            f += DF[dir]; r += DR[dir];
        }
    }
    return pinned;
}

inline Bitboard between_bb(int a, int b) { return Geometry::Between[a][b]; }
inline Bitboard line_bb(int a, int b) { return Geometry::Line[a][b]; }
inline bool aligned(int a, int b, int c) { return (line_bb(a, b) & sq_bb(c)) != 0; }

inline bool gives_check(Position& p, Move m) {
    StateInfo st;
    if (!p.make_move(m, st)) return false;
    const bool result = p.in_check(p.side);
    p.undo_move(m, st);
    return result;
}


} // namespace Scarlet

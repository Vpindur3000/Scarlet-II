#pragma once

#include "movegen.hpp"
#include "types.hpp"

// Pyrrhic numbers colours as black=0, white=1, whereas Scarlet uses the
// conventional white=0, black=1. Squares/bitboards are both A1=bit 0, so no
// byte swap is necessary at the API boundary.
#define PYRRHIC_POPCOUNT(x) (Scarlet::popcount((x)))
#define PYRRHIC_LSB(x)      (Scarlet::lsb((x)))
#define PYRRHIC_POPLSB(x)   (Scarlet::pop_lsb(*(x)))

#define PYRRHIC_PAWN_ATTACKS(sq, c) \
    (Scarlet::Attacks::Pawn[(c) ? Scarlet::WHITE : Scarlet::BLACK][(sq)])
#define PYRRHIC_KNIGHT_ATTACKS(sq)      (Scarlet::Attacks::Knight[(sq)])
#define PYRRHIC_BISHOP_ATTACKS(sq, occ) (Scarlet::Attacks::bishop_attacks((sq), (occ)))
#define PYRRHIC_ROOK_ATTACKS(sq, occ)   (Scarlet::Attacks::rook_attacks((sq), (occ)))
#define PYRRHIC_QUEEN_ATTACKS(sq, occ)  (Scarlet::Attacks::queen_attacks((sq), (occ)))
#define PYRRHIC_KING_ATTACKS(sq)        (Scarlet::Attacks::King[(sq)])

#define PYRRHIC_VALUE_PAWN   (100)
#define PYRRHIC_VALUE_MATE   (Scarlet::VALUE_MATE)
#define PYRRHIC_VALUE_DRAW   (Scarlet::VALUE_DRAW)
#define PYRRHIC_MAX_MATE_PLY (Scarlet::MAX_PLY)

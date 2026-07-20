#include "types.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    Scarlet::init();

    std::string fen = Scarlet::START_FEN;
    int depth = 5;
    bool divide = false;

    if (argc >= 2) fen = std::string(argv[1]) == "startpos" ? Scarlet::START_FEN : argv[1];
    if (argc >= 3) depth = std::atoi(argv[2]);
    if (argc >= 4 && std::string(argv[3]) == "divide") divide = true;

    Scarlet::Position pos;
    if (!pos.set_fen(fen)) {
        std::cerr << "Bad FEN\n";
        return 1;
    }

    if (divide) Scarlet::perft_divide(pos, depth);
    else {
        for (int d = 1; d <= depth; ++d)
            std::cout << "perft(" << d << ") = " << Scarlet::perft(pos, d) << "\n";
    }
    return 0;
}

#include "uci.hpp"

#include <iostream>

#if defined(_WIN32)
#include <io.h>
#define scarlet_isatty _isatty
#define scarlet_fileno _fileno
#else
#include <unistd.h>
#define scarlet_isatty isatty
#define scarlet_fileno fileno
#endif

int main() {
    // Stay silent on stdout for UCI/Arena/CuteChess compatibility. When the
    // engine is launched manually in a terminal, print a tiny helper to stderr.
    if (scarlet_isatty(scarlet_fileno(stdin))) {
        std::cerr << "Scarlet II v1.0.0 Modern-B*\n"
                  << "Type 'uci' for GUI protocol or 'help' for local commands.\n";
    }

    Scarlet::UCI::Engine engine;
    engine.loop(std::cin, std::cout);
    return 0;
}

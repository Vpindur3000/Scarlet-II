#include "uci.hpp"
#include "leela_raw.hpp"
#include "nnue_berserk.hpp"
#include "syzygy.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string_view>
#include <thread>

namespace Scarlet::UCI {

namespace {

std::vector<std::string> split(const std::string& line) {
    std::istringstream ss(line);
    std::vector<std::string> out;
    std::string token;
    while (ss >> token) out.push_back(token);
    return out;
}

std::string join(const std::vector<std::string>& v, std::size_t first, std::size_t last) {
    std::string out;
    for (std::size_t i = first; i < last && i < v.size(); ++i) {
        if (!out.empty()) out += ' ';
        out += v[i];
    }
    return out;
}

char piece_char(Piece p) {
    switch (p) {
        case W_PAWN: return 'P'; case W_KNIGHT: return 'N'; case W_BISHOP: return 'B';
        case W_ROOK: return 'R'; case W_QUEEN: return 'Q'; case W_KING: return 'K';
        case B_PAWN: return 'p'; case B_KNIGHT: return 'n'; case B_BISHOP: return 'b';
        case B_ROOK: return 'r'; case B_QUEEN: return 'q'; case B_KING: return 'k';
        default: return '.';
    }
}

std::size_t find_token(const std::vector<std::string>& tokens, const std::string& needle, std::size_t start = 0) {
    for (std::size_t i = start; i < tokens.size(); ++i)
        if (tokens[i] == needle) return i;
    return tokens.size();
}

template<typename T>
std::optional<T> parse_integer(std::string_view text) {
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) return std::nullopt;
    return value;
}

template<typename T>
bool read_integer(const std::string& text, T& value) {
    const auto parsed = parse_integer<T>(text);
    if (!parsed) return false;
    value = *parsed;
    return true;
}

bool is_go_keyword(std::string_view token) {
    return token == "searchmoves" || token == "ponder" || token == "wtime"
        || token == "btime" || token == "winc" || token == "binc"
        || token == "movestogo" || token == "depth" || token == "nodes"
        || token == "mate" || token == "movetime" || token == "infinite";
}

} // namespace

Engine::Engine() : searcher_(16) {
    Scarlet::init();
    Eval::init_backends();
    pos_.set_fen(START_FEN);
}

Engine::~Engine() {
    stop_search(true, true);
}

void Engine::stop_search(bool wait, bool suppressBestmove) {
    if (suppressBestmove) suppress_bestmove_.store(true, std::memory_order_relaxed);
    searcher_.stop();
    if (wait && search_thread_.joinable())
        search_thread_.join();
    if (wait)
        searcher_.clear_info_callback();
}

void Engine::report_result(const Search::Result& result, std::ostream& out) {
    std::lock_guard<std::mutex> lock(output_mutex_);
    if (suppress_bestmove_.load(std::memory_order_relaxed)) return;
    if (!result.debug_info.empty()) out << result.debug_info;
    const std::uint64_t nps = result.time_ms ? result.nodes * 1000ull / result.time_ms : result.nodes;
    out << "info depth " << std::max(0, result.depth);
    if (result.seldepth > 0) out << " seldepth " << result.seldepth;
    out << " score " << value_to_uci_score(result.score)
        << " time " << result.time_ms
        << " nodes " << result.nodes
        << " nps " << nps
        << " hashfull " << searcher_.hashfull();
    if (!result.search_info.empty()) {
        const bool classic = result.search_info.rfind("ClassicAB", 0) == 0;
        const bool syzygy = result.search_info.rfind("Syzygy", 0) == 0;
        out << " string ";
        if (!classic && !syzygy) out << "ModernBStar ";
        out << result.search_info;
    }
    if (!result.pv.empty()) out << " pv " << result.pv;
    out << "\n";
    if (result.lower != VALUE_NONE && result.upper != VALUE_NONE) {
        out << "info string bounds [" << result.lower << ", " << result.upper
            << "] confidence " << result.confidence
            << " source " << result.score_source
            << " degraded " << (result.degraded ? 1 : 0) << "\n";
    }
    out << "bestmove " << (result.best_move == MOVE_NONE ? std::string("0000") : move_to_uci(result.best_move)) << "\n" << std::flush;
}

void Engine::run_search_async(Position root, Search::Limits limits, std::ostream& out) {
    suppress_bestmove_.store(false, std::memory_order_relaxed);
    Search::Searcher& s = searcher_;
    search_thread_ = std::thread([this, &s, root, limits, &out]() mutable {
        const auto result = s.think(root, limits);
        report_result(result, out);
    });
}

Move parse_uci_move(Position& pos, const std::string& text) {
    MoveList legal;
    generate_legal(pos, legal);
    for (Move m : legal)
        if (move_to_uci(m) == text) return m;
    return MOVE_NONE;
}

std::string value_to_uci_score(Value v) {
    if (v >= VALUE_MATE_IN_MAX_PLY)
        return "mate " + std::to_string((VALUE_MATE - v + 1) / 2);
    if (v <= VALUE_MATED_IN_MAX_PLY)
        return "mate -" + std::to_string((VALUE_MATE + v + 1) / 2);
    return "cp " + std::to_string(v);
}

void Engine::loop(std::istream& in, std::ostream& out) {
    std::string line;
    while (std::getline(in, line)) {
        const auto tokens = split(line);
        if (tokens.empty()) continue;

        const std::string& cmd = tokens[0];
        if (cmd == "uci") {
            std::lock_guard<std::mutex> lock(output_mutex_);
            out << "id name Scarlet II v1.0.0 Modern B*\n";
            out << "id author Ruslan Fomchenkov / Scarlet contributors\n";
            out << "option name Hash type spin default 16 min 1 max 1048576\n";
            out << "option name Clear Hash type button\n";
            out << "option name SearchCore type combo default ModernBStar var ModernBStar var ClassicAB\n";
            out << "option name UseBerserkNNUE type check default true\n";
            out << "option name BerserkNNUEFile type string default assets/networks/nnue.nn\n";
            out << "option name UseLD2 type check default true\n";
            out << "option name UseLeelaRaw type check default true\n";
            out << "option name UseSyzygy type check default true\n";
            out << "option name SyzygyPath type string default <empty>\n";
            out << "option name SyzygyProbeLimit type spin default 6 min 0 max 6\n";
            out << "option name LeelaWeightsFile type string default assets/networks/policy_value.pb\n";
            out << "option name LeelaThreads type spin default 4 min 1 max 16\n";
            out << "option name LD2CacheEntries type spin default 16384 min 0 max 262144\n";
            out << "option name LD2BatchSize type spin default 4 min 1 max 32\n";
            out << "option name LD2BatchWorkers type spin default 0 min 0 max 16\n";
            out << "option name LD2RootPolicyAlways type check default true\n";
            out << "option name LD2MaxRootProbes type spin default 10 min 0 max 64\n";
            out << "option name LD2MaxFrontierProbes type spin default 64 min 0 max 512\n";
            out << "option name LD2MinPolicyForProbe type spin default 30 min 0 max 1000\n";
            out << "option name LeelaValueWeight type spin default 45 min 0 max 80\n";
            out << "option name LeelaPolicyWeight type spin default 55 min 0 max 90\n";
            out << "option name LeelaEndgameValueWeight type spin default 15 min 0 max 80\n";
            out << "option name LeelaEndgamePolicyWeight type spin default 20 min 0 max 90\n";
            out << "option name HeuristicEarlyStop type check default false\n";
            out << "option name ProofNodeLimit type spin default 65536 min 1024 max 65536\n";
            out << "option name TacticalDepth type spin default 3 min 1 max 4\n";
            out << "option name PracticalMargin type spin default 24 min 0 max 500\n";
            out << "option name MinProofExpansions type spin default 24 min 0 max 100000\n";
            out << "option name DebugModernBStar type check default false\n";
            out << "option name GuiProgress type check default true\n";
            out << "option name GuiProgressInterval type spin default 250 min 50 max 5000\n";
            out << "option name EvalBackends type string default Berserk14NNUE+LD2_raw_inprocess+SyzygyPyrrhic+HCE_sanitize\n";
            const auto& nn = NNUE::BerserkNNUE::instance().status();
            const auto& ld = Leela::RawBackend::instance().status();
            if (nn.enabled && !nn.loaded) {
                out << "info string error network BerserkNNUE path " << nn.path
                    << " reason " << nn.message << "\n"
                    << "info string warning degraded backend HCE\n";
            }
            if (Search::options().use_ld2 && ld.enabled && !ld.loaded) {
                out << "info string error network LD2 path " << ld.weights_path
                    << " reason " << ld.message << "\n"
                    << "info string warning degraded backend primary-only\n";
            }
            out << "uciok\n" << std::flush;
        } else if (cmd == "isready") {
            std::lock_guard<std::mutex> lock(output_mutex_);
            out << "readyok\n" << std::flush;
        } else if (cmd == "ucinewgame") {
            stop_search(true);
            searcher_.clear_hash();
            pos_.set_fen(START_FEN);
        } else if (cmd == "position") {
            stop_search(true);
            set_position(tokens, out);
        } else if (cmd == "go") {
            go(tokens, out);
        } else if (cmd == "ponderhit") {
            searcher_.ponderhit();
        } else if (cmd == "stop") {
            stop_search(true);
        } else if (cmd == "quit") {
            stop_search(true, true);
            break;
        } else if (cmd == "setoption") {
            stop_search(true);
            setoption(tokens, out);
        } else if (cmd == "d") {
            stop_search(true);
            std::lock_guard<std::mutex> lock(output_mutex_);
            print_board(out);
        } else if (cmd == "eval") {
            stop_search(true);
            std::lock_guard<std::mutex> lock(output_mutex_);
            out << "eval primary " << Eval::evaluate(pos_)
                << " hce " << Eval::evaluate_hce(pos_)
                << " status " << Eval::backend_status() << "\n" << std::flush;
        } else if (cmd == "backend") {
            stop_search(true);
            std::lock_guard<std::mutex> lock(output_mutex_);
            out << "info string " << Eval::backend_status() << "\n" << std::flush;
        } else if (cmd == "syzygy") {
            stop_search(true);
            std::lock_guard<std::mutex> lock(output_mutex_);
            const auto& status = Syzygy::Backend::instance().status();
            if (auto probe = Syzygy::Backend::instance().probe_root(pos_)) {
                out << "info string syzygy root wdl " << int(probe->wdl)
                    << " dtz " << probe->dtz
                    << " best " << move_to_uci(probe->best_move)
                    << " largest " << status.largest << "\n";
            } else {
                out << "info string syzygy unavailable-or-ineligible "
                    << "loaded=" << (status.loaded ? 1 : 0)
                    << " largest=" << status.largest
                    << " limit=" << status.probe_limit << "\n";
            }
            out << std::flush;
        } else if (cmd == "help" || cmd == "?") {
            std::lock_guard<std::mutex> lock(output_mutex_);
            out << "Scarlet II v1.0.0 Modern-B*\n"
                << "UCI commands: uci, isready, ucinewgame, position, go, stop, quit, setoption\n"
                << "go supports depth, movetime, nodes, clocks, infinite/ponder and searchmoves.\n"
                << "Debug commands: d, eval, backend, syzygy, leelaprobe, nnueverify <plies>, perft <n>, divide <n>\n"
                << "Useful options: SearchCore=ModernBStar/ClassicAB, DebugModernBStar, UseLD2, "
                << "LD2MaxRootProbes, LD2BatchSize, LD2BatchWorkers, SyzygyPath\n"
                << "Example: position startpos ; go depth 4\n" << std::flush;
        } else if (cmd == "leelaprobe") {
            stop_search(true);
            std::lock_guard<std::mutex> lock(output_mutex_);
            const Value fallback = Eval::evaluate(pos_);
            auto probe = Leela::RawBackend::instance().probe(pos_, fallback);
            if (probe && probe->valid) {
                out << "info string leela_raw cp " << probe->cp
                    << " wdl " << probe->wdl[0] << "/" << probe->wdl[1] << "/" << probe->wdl[2]
                    << " backend " << probe->backend;
                const int n = std::min<int>(5, probe->policy.size());
                for (int i = 0; i < n; ++i)
                    out << " p" << i << "=" << move_to_uci(probe->policy[i].move) << ":" << probe->policy[i].prior;
                out << "\n";
            } else {
                out << "info string leela unavailable " << Eval::backend_status() << "\n";
            }
            out << std::flush;
        } else if (cmd == "nnueverify") {
            stop_search(true);
            int requestedDepth = 2;
            if (tokens.size() >= 2) read_integer(tokens[1], requestedDepth);
            const int maxDepth = std::clamp(requestedDepth, 0, 4);
            auto& nnue = NNUE::BerserkNNUE::instance();
            NNUE::Accumulator rootAccumulator;
            nnue.refresh(rootAccumulator, pos_);
            std::uint64_t checked = 0;
            std::function<bool(Position&, const NNUE::Accumulator&, int)> verify =
                [&](Position& pos, const NNUE::Accumulator& accumulator, int remaining) -> bool {
                    if (!nnue.matches_full_rebuild(accumulator, pos)) return false;
                    if (remaining == 0) return true;
                    MoveList legal;
                    generate_legal(pos, legal);
                    for (Move move : legal) {
                        StateInfo state;
                        if (!pos.make_move(move, state)) continue;
                        NNUE::Accumulator child = accumulator;
                        nnue.apply_move(child, pos, move, state);
                        ++checked;
                        const bool ok = verify(pos, child, remaining - 1);
                        pos.undo_move(move, state);
                        if (!ok) return false;
                    }
                    return true;
                };
            const bool ok = rootAccumulator.valid && verify(pos_, rootAccumulator, maxDepth);
            std::lock_guard<std::mutex> lock(output_mutex_);
            out << "info string nnue incremental verify " << (ok ? "ok" : "failed")
                << " plies " << maxDepth << " positions " << checked << "\n" << std::flush;
        } else if (cmd == "perft" && tokens.size() >= 2) {
            stop_search(true);
            int requestedDepth = 0;
            if (!read_integer(tokens[1], requestedDepth)) continue;
            const int depth = std::max(0, requestedDepth);
            out << "perft(" << depth << ") = " << perft(pos_, depth) << "\n" << std::flush;
        } else if (cmd == "divide" && tokens.size() >= 2) {
            stop_search(true);
            int requestedDepth = 1;
            if (!read_integer(tokens[1], requestedDepth)) continue;
            const int depth = std::max(1, requestedDepth);
            perft_divide(pos_, depth);
        }
    }
}

void Engine::set_position(const std::vector<std::string>& tokens, std::ostream& out) {
    if (tokens.size() < 2) return;

    std::size_t movesAt = find_token(tokens, "moves", 1);
    // Build the requested position off to the side. A malformed FEN or a bad
    // move must not leave the engine at a partially applied position.
    Position next = pos_;
    bool ok = false;

    if (tokens[1] == "startpos") {
        ok = next.set_fen(START_FEN);
    } else if (tokens[1] == "fen") {
        const std::size_t fenEnd = movesAt == tokens.size() ? tokens.size() : movesAt;
        ok = next.set_fen(join(tokens, 2, fenEnd));
    }

    if (!ok) {
        out << "info string Bad position command\n" << std::flush;
        return;
    }

    if (movesAt != tokens.size()) {
        for (std::size_t i = movesAt + 1; i < tokens.size(); ++i) {
            const Move m = parse_uci_move(next, tokens[i]);
            if (m == MOVE_NONE) {
                out << "info string Illegal move in position command: " << tokens[i] << "\n" << std::flush;
                return;
            }
            StateInfo st;
            if (!next.make_move(m, st)) {
                out << "info string Failed to make move: " << tokens[i] << "\n" << std::flush;
                return;
            }
        }
    }

    pos_ = std::move(next);
}

void Engine::go(const std::vector<std::string>& tokens, std::ostream& out) {
    // Stop and join any previous analysis search before starting a new one.
    stop_search(true);
    // Arm the stop flag before an asynchronous worker is launched. Resetting
    // inside the worker races with an immediate UCI `stop` and can otherwise
    // turn `go infinite; stop` into an unjoinable search.
    searcher_.reset_stop();
    searcher_.set_info_callback([this, &out](std::string_view line) {
        std::lock_guard<std::mutex> lock(output_mutex_);
        out << line << std::flush;
    });

    Search::Limits limits;
    limits.depth = 6;

    int wtime = 0, btime = 0, winc = 0, binc = 0, movesToGo = 0;
    int requestedMoveTime = 0;
    bool explicitDepth = false;
    bool explicitMoveTime = false;
    bool invalidArgument = false;

    for (std::size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == "searchmoves") {
            const std::size_t firstMove = i + 1;
            while (i + 1 < tokens.size() && !is_go_keyword(tokens[i + 1])) {
                const Move move = parse_uci_move(pos_, tokens[++i]);
                if (move == MOVE_NONE) {
                    invalidArgument = true;
                    continue;
                }
                const TTMove core = move_to_tt(move);
                if (std::find(limits.searchmoves.begin(), limits.searchmoves.end(), core)
                        == limits.searchmoves.end())
                    limits.searchmoves.push_back(core);
            }
            if (i + 1 == firstMove) invalidArgument = true;
        }
        else if (tokens[i] == "depth" && i + 1 < tokens.size()) {
            int value = 0;
            if (read_integer(tokens[++i], value)) {
                limits.depth = std::clamp(value, 0, 128);
                explicitDepth = true;
            } else invalidArgument = true;
        }
        else if (tokens[i] == "movetime" && i + 1 < tokens.size()) {
            int value = 0;
            if (read_integer(tokens[++i], value)) {
                requestedMoveTime = std::max(0, value);
                limits.movetime_ms = requestedMoveTime;
                explicitMoveTime = true;
            } else invalidArgument = true;
        }
        else if (tokens[i] == "nodes" && i + 1 < tokens.size()) {
            std::uint64_t value = 0;
            if (read_integer(tokens[++i], value)) limits.nodes = value;
            else invalidArgument = true;
        }
        else if (tokens[i] == "movestogo" && i + 1 < tokens.size()) {
            int value = 0;
            if (read_integer(tokens[++i], value) && value > 0) movesToGo = value;
            else invalidArgument = true;
        }
        else if (tokens[i] == "mate" && i + 1 < tokens.size()) {
            int value = 0;
            if (read_integer(tokens[++i], value) && value > 0) {
                // UCI mate N is a search request, not a promise. Two plies per
                // move plus the mating ply is a useful verification ceiling.
                limits.depth = std::clamp(value * 2, 1, 128);
                explicitDepth = true;
            } else invalidArgument = true;
        }
        else if (tokens[i] == "infinite") limits.infinite = true;
        else if (tokens[i] == "ponder") { limits.infinite = true; limits.ponder = true; }
        else if (tokens[i] == "wtime" && i + 1 < tokens.size()) {
            int value = 0;
            if (read_integer(tokens[++i], value)) wtime = std::max(0, value);
            else invalidArgument = true;
        }
        else if (tokens[i] == "btime" && i + 1 < tokens.size()) {
            int value = 0;
            if (read_integer(tokens[++i], value)) btime = std::max(0, value);
            else invalidArgument = true;
        }
        else if (tokens[i] == "winc" && i + 1 < tokens.size()) {
            int value = 0;
            if (read_integer(tokens[++i], value)) winc = std::max(0, value);
            else invalidArgument = true;
        }
        else if (tokens[i] == "binc" && i + 1 < tokens.size()) {
            int value = 0;
            if (read_integer(tokens[++i], value)) binc = std::max(0, value);
            else invalidArgument = true;
        }
    }

    if (invalidArgument) {
        std::lock_guard<std::mutex> lock(output_mutex_);
        out << "info string ignored go command with invalid argument\n" << std::flush;
        searcher_.clear_info_callback();
        return;
    }

    if (limits.infinite) {
        // Arena analysis sends `go infinite` and expects the engine to keep
        // emitting info until `stop`. Use a high internal proof target, but do
        // not let this fake a final depth; progress depth is time/work based.
        limits.depth = explicitDepth ? limits.depth : 64;
        limits.movetime_ms = 0;
        if (limits.ponder) {
            if (explicitMoveTime) {
                limits.ponderhit_movetime_ms = std::max(0, requestedMoveTime);
            } else {
                const int clock = pos_.side == WHITE ? wtime : btime;
                const int inc = pos_.side == WHITE ? winc : binc;
                if (clock > 0)
                    limits.ponderhit_movetime_ms = std::max(
                        20, clock / std::max(1, movesToGo > 0 ? movesToGo : 35) + inc / 2);
            }
        }
    } else if (!explicitMoveTime) {
        const int clock = pos_.side == WHITE ? wtime : btime;
        const int inc = pos_.side == WHITE ? winc : binc;
        if (clock > 0) {
            // Uncertainty-aware search later extends inside Modern B*; this is the safe base slice.
            limits.movetime_ms = std::max(
                20, clock / std::max(1, movesToGo > 0 ? movesToGo : 35) + inc / 2);
        }
    }
    if (limits.movetime_ms > 0 && !explicitDepth) limits.depth = 32;

    run_search_async(pos_, limits, out);
}

namespace {

bool parse_bool_option(const std::string& v) {
    std::string x = v;
    std::transform(x.begin(), x.end(), x.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return x == "true" || x == "1" || x == "yes" || x == "on";
}

} // namespace

void Engine::setoption(const std::vector<std::string>& tokens, std::ostream& out) {
    const auto nameAt = find_token(tokens, "name", 1);
    const auto valueAt = find_token(tokens, "value", 1);
    if (nameAt == tokens.size()) return;

    const std::string name = valueAt == tokens.size() ? join(tokens, nameAt + 1, tokens.size())
                                                       : join(tokens, nameAt + 1, valueAt);
    const std::string value = valueAt == tokens.size() ? std::string{} : join(tokens, valueAt + 1, tokens.size());

    if (name == "Hash" && !value.empty()) {
        std::uint64_t parsed = 0;
        if (read_integer(value, parsed)) {
            const std::size_t requestedMb = std::size_t(std::clamp<std::uint64_t>(parsed, 1, 1048576));
            try {
                searcher_.set_hash(requestedMb);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(output_mutex_);
                out << "info string failed to set Hash: " << e.what() << "\n" << std::flush;
            }
        }
    } else if (name == "Clear Hash") {
        searcher_.clear_hash();
        Leela::RawBackend::instance().clear_cache();
    } else if (name == "SearchCore" && !value.empty()) {
        std::string v = value;
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        Search::set_modern_bstar_enabled(v != "classicab" && v != "pureab" && v != "classic");
    } else if (name == "UseBerserkNNUE" && !value.empty()) {
        Eval::set_berserk_nnue_enabled(parse_bool_option(value));
    } else if (name == "BerserkNNUEFile" && !value.empty()) {
        if (!Eval::load_berserk_nnue(value)) {
            const auto& status = NNUE::BerserkNNUE::instance().status();
            std::lock_guard<std::mutex> lock(output_mutex_);
            out << "info string error network BerserkNNUE path " << status.path
                << " reason " << status.message << "\n"
                << "info string warning degraded backend HCE\n" << std::flush;
        }
    } else if ((name == "UseLD2" || name == "UseLeelaRaw" || name == "UseLeelaOnnx") && !value.empty()) {
        const bool on = parse_bool_option(value);
        Search::options().use_ld2 = on;
        Leela::RawBackend::instance().set_enabled(on);
    } else if (name == "UseSyzygy" && !value.empty()) {
        Syzygy::Backend::instance().set_enabled(parse_bool_option(value));
    } else if (name == "SyzygyPath" && !value.empty()) {
        const bool loaded = Syzygy::Backend::instance().set_path(value);
        const auto& status = Syzygy::Backend::instance().status();
        std::lock_guard<std::mutex> lock(output_mutex_);
        out << "info string SyzygyPath " << (loaded ? "loaded" : "not loaded")
            << " largest " << status.largest << "\n" << std::flush;
    } else if (name == "SyzygyProbeLimit" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Syzygy::Backend::instance().set_probe_limit(parsed);
    } else if (name == "LeelaWeightsFile" && !value.empty()) {
        if (!Leela::RawBackend::instance().load(value)) {
            const auto& status = Leela::RawBackend::instance().status();
            std::lock_guard<std::mutex> lock(output_mutex_);
            out << "info string error network LD2 path " << status.weights_path
                << " reason " << status.message << "\n"
                << "info string warning degraded backend primary-only\n" << std::flush;
        }
    } else if (name == "LeelaThreads" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Leela::RawBackend::instance().set_threads(std::clamp(parsed, 1, 16));
    } else if (name == "LD2CacheEntries" && !value.empty()) {
        std::uint64_t parsed = 0;
        if (read_integer(value, parsed))
            Leela::RawBackend::instance().set_cache_capacity(std::min<std::uint64_t>(parsed, 262144));
    } else if (name == "LD2BatchSize" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Search::options().ld2_batch_size = std::clamp(parsed, 1, 32);
    } else if (name == "LD2BatchWorkers" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Leela::RawBackend::instance().set_batch_workers(parsed);
    } else if (name == "LD2RootPolicyAlways" && !value.empty()) {
        Search::options().ld2_root_policy_always = parse_bool_option(value);
    } else if (name == "LD2MaxRootProbes" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Search::options().ld2_max_root_probes = std::clamp(parsed, 0, 64);
    } else if (name == "LD2MaxFrontierProbes" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Search::options().ld2_max_frontier_probes = std::clamp(parsed, 0, 512);
    } else if (name == "LD2MinPolicyForProbe" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Search::options().ld2_min_policy_permille = std::clamp(parsed, 0, 1000);
    } else if (name == "LeelaValueWeight" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Search::options().leela_value_weight = std::clamp(parsed, 0, 80);
    } else if (name == "LeelaPolicyWeight" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Search::options().leela_policy_weight = std::clamp(parsed, 0, 90);
    } else if (name == "LeelaEndgameValueWeight" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Search::options().leela_endgame_value_weight = std::clamp(parsed, 0, 80);
    } else if (name == "LeelaEndgamePolicyWeight" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Search::options().leela_endgame_policy_weight = std::clamp(parsed, 0, 90);
    } else if (name == "HeuristicEarlyStop" && !value.empty()) {
        Search::options().heuristic_early_stop = parse_bool_option(value);
    } else if (name == "ProofNodeLimit" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Search::options().proof_node_limit = std::clamp(parsed, 1024, 65536);
    } else if (name == "TacticalDepth" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Search::options().tactical_depth = std::clamp(parsed, 1, 4);
    } else if (name == "PracticalMargin" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Search::options().practical_margin = std::clamp(parsed, 0, 500);
    } else if (name == "MinProofExpansions" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Search::options().min_proof_expansions = std::clamp(parsed, 0, 100000);
    } else if (name == "DebugModernBStar" && !value.empty()) {
        Search::options().debug_modern_bstar = parse_bool_option(value);
    } else if (name == "GuiProgress" && !value.empty()) {
        Search::options().gui_progress = parse_bool_option(value);
    } else if (name == "GuiProgressInterval" && !value.empty()) {
        int parsed = 0;
        if (read_integer(value, parsed)) Search::options().gui_progress_interval_ms = std::clamp(parsed, 50, 5000);
    } else if (name == "Lc0Path" || name == "LeelaBackend" || name == "LeelaProbeNodes" ||
               name == "LeelaProbeTimeout") {
        // Deprecated no-op options are accepted so old GUIs/scripts do not break.
    }
}

void Engine::print_board(std::ostream& out) const {
    for (int r = 7; r >= 0; --r) {
        out << r + 1 << "  ";
        for (int f = 0; f < 8; ++f)
            out << piece_char(pos_.board[r * 8 + f]) << ' ';
        out << "\n";
    }
    out << "\n   a b c d e f g h\n";
    out << "side " << (pos_.side == WHITE ? "white" : "black")
        << " key " << pos_.key
        << " pawnKey " << pos_.pawnKey << "\n" << std::flush;
}

} // namespace Scarlet::UCI

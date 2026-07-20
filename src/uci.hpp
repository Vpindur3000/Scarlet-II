#pragma once

#include "search.hpp"

#include <iosfwd>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace Scarlet::UCI {

class Engine {
public:
    Engine();
    ~Engine();
    void loop(std::istream& in, std::ostream& out);

private:
    Position pos_{};
    Search::Searcher searcher_;
    std::thread search_thread_;
    std::mutex output_mutex_;
    std::atomic<bool> suppress_bestmove_{false};

    void stop_search(bool wait, bool suppressBestmove = false);
    void report_result(const Search::Result& result, std::ostream& out);
    void run_search_async(Position root, Search::Limits limits, std::ostream& out);

    void set_position(const std::vector<std::string>& tokens, std::ostream& out);
    void go(const std::vector<std::string>& tokens, std::ostream& out);
    void setoption(const std::vector<std::string>& tokens, std::ostream& out);
    void print_board(std::ostream& out) const;
};

Move parse_uci_move(Position& pos, const std::string& text);
std::string value_to_uci_score(Value v);

} // namespace Scarlet::UCI

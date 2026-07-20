#pragma once

#include "types.hpp"

#include <array>
#include <cstdint>
#include <list>
#include <optional>
#include <string>
#include <string_view>
#include <stop_token>
#include <unordered_map>
#include <vector>

namespace Scarlet::Leela {

struct PolicyEntry {
    Move move = MOVE_NONE;
    float prior = 0.0f;
    int plane = -1;
};

struct RawProbeResult {
    bool valid = false;
    Value cp = VALUE_NONE;
    std::array<float, 3> wdl{0.0f, 0.0f, 0.0f}; // W, D, L from side-to-move POV.
    std::vector<PolicyEntry> policy;
    std::string backend;
};

struct RawProbeRequest {
    Position position{};
    Value fallback_cp = VALUE_ZERO;
};

struct RawStatus {
    bool enabled = true;
    bool loaded = false;
    std::string weights_path;
    std::uint64_t probes = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t cache_evictions = 0;
    std::uint64_t cache_entries = 0;
    std::uint64_t cache_capacity = 16384;
    std::uint64_t batch_calls = 0;
    std::uint64_t batch_positions = 0;
    std::uint64_t policy_queries = 0;
    std::uint64_t policy_hits = 0;
    std::uint64_t total_us = 0;
    std::uint64_t failures = 0;
    std::uint64_t workspace_reallocations = 0;
    std::uint64_t weight_bytes = 0;
    int channels = 0;
    int residual_blocks = 0;
    int policy_planes = 0;
    int value_planes = 0;
    int threads = 1;
    int batch_workers = 0;
    bool openmp = false;
    std::string message;
};

class RawBackend final {
public:
    static RawBackend& instance();

    RawBackend(const RawBackend&) = delete;
    RawBackend& operator=(const RawBackend&) = delete;
    ~RawBackend() = default;

    bool try_load_default();
    bool load(std::string_view path);
    void set_enabled(bool enabled);
    void set_threads(int threads);
    void set_batch_workers(int workers);
    void set_cache_capacity(std::size_t entries);
    void clear_cache();

    [[nodiscard]] const RawStatus& status() const { return status_; }
    [[nodiscard]] std::optional<RawProbeResult> probe(
        const Position& pos, Value fallbackCp, std::stop_token stop = {});
    [[nodiscard]] std::vector<std::optional<RawProbeResult>> probe_batch(
        const std::vector<RawProbeRequest>& requests, std::stop_token stop = {});
    [[nodiscard]] std::optional<RawProbeResult> cached_probe(const Position& pos);
    [[nodiscard]] float cached_policy_prior(const Position& pos, Move move);
    [[nodiscard]] std::optional<RawProbeResult> scalar_reference_probe(
        const Position& pos, Value fallbackCp);
    // Canonical LC0 convolutional-policy plane. Public so the mapping can be
    // parity-tested without loading a network.
    [[nodiscard]] static int canonical_policy_plane(const Position& pos, Move move);

private:
    RawBackend() = default;

public:
    struct Layer {
        std::vector<float> v;
    };
    struct ConvBlock {
        std::vector<float> weights;
        std::vector<float> biases;
        int in_channels = 0;
        int out_channels = 0;
        int kernel = 0;
    };
    struct SEUnit {
        std::vector<float> w1, b1, w2, b2;
    };
    struct ResidualBlock {
        ConvBlock conv1, conv2;
        SEUnit se;
        bool has_se = false;
    };
    struct Network {
        ConvBlock input;
        std::vector<ResidualBlock> residual;
        ConvBlock policy1;
        ConvBlock policy;
        ConvBlock value;
        std::vector<float> ip1_val_w, ip1_val_b;
        std::vector<float> ip2_val_w, ip2_val_b;
        int channels = 0;
        int policy_planes = 0;
        int value_planes = 0;
        bool valid = false;
    };

private:
    Network net_;

    [[nodiscard]] std::optional<RawProbeResult> run_network(
        const Position& pos, Value fallbackCp, std::stop_token stop = {});
    [[nodiscard]] std::vector<std::optional<RawProbeResult>> run_network_batch(
        const std::vector<RawProbeRequest>& requests, std::stop_token stop = {});
    [[nodiscard]] std::vector<float> encode_planes(const Position& pos) const;
    void encode_planes_into(const Position& pos, float* planes) const;
    [[nodiscard]] float raw_policy_prior_from_probe(const RawProbeResult& probe, Move move) const;
    void cache_result(Key key, RawProbeResult result);

    RawStatus status_{};
    struct CacheEntry {
        RawProbeResult result;
        std::list<Key>::iterator lru;
    };
    std::unordered_map<Key, CacheEntry> cache_;
    std::list<Key> cache_lru_; // front = LRU, back = MRU
    std::size_t cache_capacity_ = 16384;
    int batch_workers_requested_ = 0; // 0 = auto from LeelaThreads.
};

} // namespace Scarlet::Leela

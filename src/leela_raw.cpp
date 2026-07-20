#include "leela_raw.hpp"
#include "runtime_paths.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <chrono>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(SCARLET_USE_OPENMP)
#include <omp.h>
#endif

#ifdef SCARLET_USE_ZLIB
#include <zlib.h>
#endif

namespace Scarlet::Leela {
namespace {

constexpr int kSquares = 64;
constexpr int kInputPlanes = 112;
constexpr int kPlanesPerBoard = 13;
constexpr int kMoveHistory = 8;
constexpr int kAuxPlaneBase = kPlanesPerBoard * kMoveHistory; // 104
constexpr std::uint32_t kWeightMagic = 0x1c0;
constexpr float kEpsilon = 1e-5f;

// One persistent, coarse-grained kernel team processes complete tensor layers.
// The option historically named LD2BatchWorkers now controls this team size.
thread_local int tls_inference_threads = 0;

struct BatchScratch {
    std::vector<float> b0, b1, b2, b3;
    std::vector<float> padded;
    std::vector<float> pool, fc1, scale, hidden, valueLogits;
};
// UCI guarantees one joined search at a time. Keeping the large workspace at
// backend lifetime (rather than thread lifetime) preserves warmup across the
// short-lived worker thread created for every `go`.
BatchScratch persistent_batch_scratch;

[[maybe_unused]] int inference_omp_threads() {
#if defined(SCARLET_USE_OPENMP)
    return tls_inference_threads > 0 ? tls_inference_threads : omp_get_max_threads();
#else
    return 1;
#endif
}

void warm_kernel_team(int threads) {
#if defined(SCARLET_USE_OPENMP)
    int participants = 0;
#pragma omp parallel num_threads(threads)
    {
#pragma omp atomic update
        participants += 1;
    }
    (void)participants;
#else
    (void)threads;
#endif
}

void reserve_batch_workspace(const RawBackend::Network& net, int maxBatch) {
    BatchScratch& scratch = persistent_batch_scratch;
    const std::size_t batch = std::size_t(maxBatch);
    const int maxChannels = std::max({kInputPlanes, net.channels, net.policy_planes,
                                     net.value_planes});
    int maxSeHidden = 0;
    for (const auto& residual : net.residual)
        maxSeHidden = std::max(maxSeHidden, int(residual.se.b1.size()));
    scratch.b0.reserve(batch * kInputPlanes * kSquares);
    scratch.b1.reserve(batch * std::size_t(maxChannels) * kSquares);
    scratch.b2.reserve(batch * std::size_t(maxChannels) * kSquares);
    scratch.b3.reserve(batch * std::size_t(maxChannels) * kSquares);
    scratch.padded.reserve(batch * std::size_t(std::max(kInputPlanes, net.channels)) * 100);
    scratch.pool.reserve(batch * std::size_t(net.channels));
    scratch.fc1.reserve(batch * std::size_t(std::max(1, maxSeHidden)));
    scratch.scale.reserve(batch * std::size_t(2 * net.channels));
    scratch.hidden.reserve(batch * net.ip1_val_b.size());
    scratch.valueLogits.reserve(batch * 3);
}

Value clamp_cp(int v) {
    return std::clamp(v, -VALUE_MATE_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1);
}

float relu(float x) { return x > 0.0f ? x : 0.0f; }
float sigmoid(float x) {
    if (x >= 40.0f) return 1.0f;
    if (x <= -40.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

std::string read_binary_file(std::string_view path) {
    std::ifstream in(std::string(path), std::ios::binary);
    if (!in) throw std::runtime_error("cannot open weights file");
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool ends_with(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}

#ifdef SCARLET_USE_ZLIB
std::string read_gzip_file(std::string_view path) {
    gzFile f = gzopen(std::string(path).c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open gzip weights");
    std::string out;
    out.resize(8 * 1024 * 1024);
    int used = 0;
    for (;;) {
        if (used == int(out.size())) out.resize(out.size() * 2);
        const int n = gzread(f, out.data() + used, unsigned(out.size() - used));
        if (n < 0) {
            int err = 0;
            const char* msg = gzerror(f, &err);
            gzclose(f);
            throw std::runtime_error(msg ? msg : "gzip read failed");
        }
        used += n;
        if (n == 0) break;
    }
    gzclose(f);
    out.resize(used);
    return out;
}
#endif

std::string read_weights_file(std::string_view path) {
    if (ends_with(path, ".gz")) {
#ifdef SCARLET_USE_ZLIB
        return read_gzip_file(path);
#else
        throw std::runtime_error("gzip support is not built; use uncompressed policy_value.pb");
#endif
    }
    return read_binary_file(path);
}

struct Field {
    int id = 0;
    int wire = 0;
    std::string_view data{};
    std::uint64_t var = 0;
    std::uint32_t fixed32 = 0;
};

std::uint64_t read_varint(const std::string_view s, std::size_t& i) {
    std::uint64_t value = 0;
    int shift = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i++]);
        value |= std::uint64_t(c & 0x7f) << shift;
        if ((c & 0x80) == 0) return value;
        shift += 7;
        if (shift > 63) throw std::runtime_error("bad varint");
    }
    throw std::runtime_error("truncated varint");
}

std::uint32_t read_fixed32(std::string_view s, std::size_t& i) {
    if (i + 4 > s.size()) throw std::runtime_error("truncated fixed32");
    std::uint32_t v = (std::uint32_t(static_cast<unsigned char>(s[i + 0])) << 0) |
                      (std::uint32_t(static_cast<unsigned char>(s[i + 1])) << 8) |
                      (std::uint32_t(static_cast<unsigned char>(s[i + 2])) << 16) |
                      (std::uint32_t(static_cast<unsigned char>(s[i + 3])) << 24);
    i += 4;
    return v;
}

float fixed32_to_float(std::uint32_t u) {
    float f;
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

std::vector<Field> parse_fields(std::string_view s) {
    std::vector<Field> out;
    std::size_t i = 0;
    while (i < s.size()) {
        const std::uint64_t tag = read_varint(s, i);
        Field f;
        f.id = int(tag >> 3);
        f.wire = int(tag & 7);
        switch (f.wire) {
            case 0:
                f.var = read_varint(s, i);
                break;
            case 1:
                if (i + 8 > s.size()) throw std::runtime_error("truncated fixed64");
                f.data = s.substr(i, 8);
                i += 8;
                break;
            case 2: {
                const std::uint64_t len = read_varint(s, i);
                if (i + len > s.size()) throw std::runtime_error("truncated len field");
                f.data = s.substr(i, std::size_t(len));
                i += std::size_t(len);
                break;
            }
            case 5:
                f.fixed32 = read_fixed32(s, i);
                break;
            default:
                throw std::runtime_error("unsupported protobuf wire type");
        }
        out.push_back(f);
    }
    return out;
}

std::string_view first_len_field(std::string_view msg, int id) {
    for (const auto& f : parse_fields(msg))
        if (f.id == id && f.wire == 2) return f.data;
    return {};
}

std::vector<std::string_view> all_len_fields(std::string_view msg, int id) {
    std::vector<std::string_view> r;
    for (const auto& f : parse_fields(msg))
        if (f.id == id && f.wire == 2) r.push_back(f.data);
    return r;
}

std::vector<float> decode_layer(std::string_view msg) {
    float minv = 0.0f, maxv = 0.0f;
    std::string_view params;
    for (const auto& f : parse_fields(msg)) {
        if (f.id == 1 && f.wire == 5) minv = fixed32_to_float(f.fixed32);
        else if (f.id == 2 && f.wire == 5) maxv = fixed32_to_float(f.fixed32);
        else if (f.id == 3 && f.wire == 2) params = f.data;
    }
    std::vector<float> out;
    if (params.empty()) return out;
    if ((params.size() & 1u) != 0)
        throw std::runtime_error("LINEAR16 layer has odd byte count");
    if (!std::isfinite(minv) || !std::isfinite(maxv) || maxv < minv)
        throw std::runtime_error("invalid LINEAR16 range");
    out.resize(params.size() / 2);
    const float range = maxv - minv;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const auto lo = static_cast<unsigned char>(params[2 * i]);
        const auto hi = static_cast<unsigned char>(params[2 * i + 1]);
        const std::uint16_t q = std::uint16_t(lo | (hi << 8));
        out[i] = minv + (float(q) / 65535.0f) * range;
    }
    return out;
}

RawBackend::ConvBlock decode_convblock(std::string_view msg, int knownIn, int knownOut, int knownKernel) {
    RawBackend::ConvBlock cb;
    std::vector<float> bnGammas, bnBetas, bnMeans, bnStd;
    for (const auto& f : parse_fields(msg)) {
        if (f.wire != 2) continue;
        switch (f.id) {
            case 1: cb.weights = decode_layer(f.data); break;
            case 2: cb.biases = decode_layer(f.data); break;
            case 3: bnMeans = decode_layer(f.data); break;
            case 4: bnStd = decode_layer(f.data); break;
            case 5: bnGammas = decode_layer(f.data); break;
            case 6: bnBetas = decode_layer(f.data); break;
            default: break;
        }
    }
    if (knownOut > 0) cb.out_channels = knownOut;
    else if (!cb.biases.empty()) cb.out_channels = int(cb.biases.size());
    else if (!bnMeans.empty()) cb.out_channels = int(bnMeans.size());
    else if (!bnGammas.empty()) cb.out_channels = int(bnGammas.size());
    else cb.out_channels = 0;

    if (knownKernel > 0) cb.kernel = knownKernel;
    else cb.kernel = 1;

    if (knownIn > 0) cb.in_channels = knownIn;
    else if (cb.out_channels > 0 && !cb.weights.empty()) {
        const int div = cb.out_channels * cb.kernel * cb.kernel;
        cb.in_channels = div > 0 ? int(cb.weights.size()) / div : 0;
    }

    if (cb.out_channels == 0 && knownIn > 0 && knownKernel > 0 && !cb.weights.empty()) {
        cb.out_channels = int(cb.weights.size()) / (knownIn * knownKernel * knownKernel);
    }
    if (cb.biases.empty() && cb.out_channels > 0) cb.biases.assign(cb.out_channels, 0.0f);

    if (!bnMeans.empty()) {
        if (bnBetas.empty()) bnBetas.assign(bnMeans.size(), 0.0f);
        if (bnGammas.empty()) bnGammas.assign(bnMeans.size(), 1.0f);
        if (cb.biases.empty()) cb.biases.assign(bnMeans.size(), 0.0f);
        if (bnStd.size() == bnMeans.size() && bnGammas.size() == bnMeans.size() && bnBetas.size() == bnMeans.size()) {
            for (std::size_t o = 0; o < bnStd.size(); ++o) {
                bnGammas[o] *= 1.0f / std::sqrt(bnStd[o] + kEpsilon);
                bnMeans[o] -= cb.biases[o];
            }
            const std::size_t inputs = cb.out_channels ? cb.weights.size() / cb.out_channels : 0;
            for (std::size_t o = 0; o < std::size_t(cb.out_channels); ++o) {
                for (std::size_t c = 0; c < inputs; ++c)
                    cb.weights[o * inputs + c] *= bnGammas[o];
                cb.biases[o] = -bnGammas[o] * bnMeans[o] + bnBetas[o];
            }
        }
    }
    return cb;
}

RawBackend::SEUnit decode_se(std::string_view msg) {
    RawBackend::SEUnit se;
    for (const auto& f : parse_fields(msg)) {
        if (f.wire != 2) continue;
        if (f.id == 1) se.w1 = decode_layer(f.data);
        else if (f.id == 2) se.b1 = decode_layer(f.data);
        else if (f.id == 3) se.w2 = decode_layer(f.data);
        else if (f.id == 4) se.b2 = decode_layer(f.data);
    }
    return se;
}

void validate_conv(const RawBackend::ConvBlock& cb, std::string_view name,
                   int expectedIn, int expectedOut, int expectedKernel) {
    if (cb.in_channels != expectedIn || cb.out_channels != expectedOut || cb.kernel != expectedKernel)
        throw std::runtime_error(std::string(name) + " has invalid dimensions");
    const std::size_t expectedWeights = std::size_t(expectedIn) * expectedOut
                                      * expectedKernel * expectedKernel;
    if (cb.weights.size() != expectedWeights || cb.biases.size() != std::size_t(expectedOut))
        throw std::runtime_error(std::string(name) + " has invalid buffer sizes");
}

void validate_fc(std::string_view name, const std::vector<float>& weights,
                 const std::vector<float>& biases, std::size_t inputs, std::size_t outputs) {
    if (biases.size() != outputs || weights.size() != inputs * outputs)
        throw std::runtime_error(std::string(name) + " has invalid buffer sizes");
}

#if defined(__AVX2__)
inline __m256 fmadd8(__m256 a, __m256 b, __m256 c) {
#if defined(__FMA__)
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}
#endif

void conv3_forward(const std::vector<float>& in, int inC, int outC,
                   const std::vector<float>& w, const std::vector<float>& b,
                   std::vector<float>& out, bool reluOut) {
    out.assign(std::size_t(outC) * kSquares, 0.0f);
#if defined(__AVX2__)
    // Fast path for the 8x8 LC0 board tensor.  The previous scalar version spent
    // most time recomputing border checks inside the innermost convolution loops.
    // Padding once lets every 3x3 tap become eight contiguous float loads per rank.
    std::vector<float> padded(std::size_t(inC) * 100, 0.0f); // 10x10 per channel.
    for (int ic = 0; ic < inC; ++ic) {
        const float* src = in.data() + std::size_t(ic) * kSquares;
        float* dst = padded.data() + std::size_t(ic) * 100;
        for (int r = 0; r < 8; ++r)
            std::memcpy(dst + (r + 1) * 10 + 1, src + r * 8, 8 * sizeof(float));
    }

    const __m256 zero = _mm256_setzero_ps();
#if defined(SCARLET_USE_OPENMP)
#pragma omp parallel for schedule(static) num_threads(inference_omp_threads())
#endif
    for (int oc = 0; oc < outC; ++oc) {
        float* dst = out.data() + std::size_t(oc) * kSquares;
        const __m256 bias = _mm256_set1_ps(b.empty() ? 0.0f : b[oc]);
        for (int r = 0; r < 8; ++r)
            _mm256_storeu_ps(dst + r * 8, bias);

        const float* wb = w.data() + std::size_t(oc) * inC * 9;
        for (int ic = 0; ic < inC; ++ic) {
            const float* src = padded.data() + std::size_t(ic) * 100;
            const float* wk = wb + ic * 9;
            for (int kr = 0; kr < 3; ++kr) {
                for (int kc = 0; kc < 3; ++kc) {
                    const __m256 ww = _mm256_set1_ps(wk[kr * 3 + kc]);
                    for (int r = 0; r < 8; ++r) {
                        float* d = dst + r * 8;
                        const float* s = src + (r + kr) * 10 + kc;
                        __m256 acc = _mm256_loadu_ps(d);
                        const __m256 x = _mm256_loadu_ps(s);
                        acc = fmadd8(ww, x, acc);
                        _mm256_storeu_ps(d, acc);
                    }
                }
            }
        }
        if (reluOut) {
            for (int r = 0; r < 8; ++r) {
                float* d = dst + r * 8;
                _mm256_storeu_ps(d, _mm256_max_ps(_mm256_loadu_ps(d), zero));
            }
        }
    }
#else
    const int kernelSize = 9;
    for (int oc = 0; oc < outC; ++oc) {
        float* dst = out.data() + oc * kSquares;
        const float* wb = w.data() + std::size_t(oc) * inC * kernelSize;
        for (int sq = 0; sq < kSquares; ++sq) dst[sq] = b.empty() ? 0.0f : b[oc];
        for (int ic = 0; ic < inC; ++ic) {
            const float* src = in.data() + ic * kSquares;
            const float* wk = wb + ic * kernelSize;
            for (int r = 0; r < 8; ++r) {
                for (int f = 0; f < 8; ++f) {
                    float sum = 0.0f;
                    int ki = 0;
                    for (int dr = -1; dr <= 1; ++dr) {
                        const int rr = r + dr;
                        for (int df = -1; df <= 1; ++df, ++ki) {
                            const int ff = f + df;
                            if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8)
                                sum += wk[ki] * src[rr * 8 + ff];
                        }
                    }
                    dst[r * 8 + f] += sum;
                }
            }
        }
        if (reluOut) for (int sq = 0; sq < kSquares; ++sq) dst[sq] = relu(dst[sq]);
    }
#endif
}

void conv1_forward(const std::vector<float>& in, int inC, int outC,
                   const std::vector<float>& w, const std::vector<float>& b,
                   std::vector<float>& out, bool reluOut) {
    out.assign(std::size_t(outC) * kSquares, 0.0f);
#if defined(__AVX2__)
    const __m256 zero = _mm256_setzero_ps();
#if defined(SCARLET_USE_OPENMP)
#pragma omp parallel for schedule(static) num_threads(inference_omp_threads())
#endif
    for (int oc = 0; oc < outC; ++oc) {
        float* dst = out.data() + std::size_t(oc) * kSquares;
        const __m256 bias = _mm256_set1_ps(b.empty() ? 0.0f : b[oc]);
        for (int r = 0; r < 8; ++r)
            _mm256_storeu_ps(dst + r * 8, bias);
        const float* ww = w.data() + std::size_t(oc) * inC;
        for (int ic = 0; ic < inC; ++ic) {
            const __m256 wc = _mm256_set1_ps(ww[ic]);
            const float* src = in.data() + std::size_t(ic) * kSquares;
            for (int r = 0; r < 8; ++r) {
                float* d = dst + r * 8;
                __m256 acc = _mm256_loadu_ps(d);
                const __m256 x = _mm256_loadu_ps(src + r * 8);
                acc = fmadd8(wc, x, acc);
                _mm256_storeu_ps(d, acc);
            }
        }
        if (reluOut) {
            for (int r = 0; r < 8; ++r) {
                float* d = dst + r * 8;
                _mm256_storeu_ps(d, _mm256_max_ps(_mm256_loadu_ps(d), zero));
            }
        }
    }
#else
    for (int oc = 0; oc < outC; ++oc) {
        float* dst = out.data() + oc * kSquares;
        for (int sq = 0; sq < kSquares; ++sq) dst[sq] = b.empty() ? 0.0f : b[oc];
        const float* ww = w.data() + std::size_t(oc) * inC;
        for (int ic = 0; ic < inC; ++ic) {
            const float wc = ww[ic];
            const float* src = in.data() + ic * kSquares;
            for (int sq = 0; sq < kSquares; ++sq) dst[sq] += wc * src[sq];
        }
        if (reluOut) for (int sq = 0; sq < kSquares; ++sq) dst[sq] = relu(dst[sq]);
    }
#endif
}

void fc_forward(const std::vector<float>& in, const std::vector<float>& w,
                const std::vector<float>& b, int outSize,
                std::vector<float>& out, bool reluOut) {
    const int inSize = int(in.size());
    out.assign(outSize, 0.0f);
    for (int o = 0; o < outSize; ++o) {
        const float* ww = w.data() + std::size_t(o) * inSize;
        float sum = b.empty() ? 0.0f : b[o];
        for (int i = 0; i < inSize; ++i) sum += ww[i] * in[i];
        out[o] = reluOut ? relu(sum) : sum;
    }
}

// True NCHW batch kernels. A layer is traversed once for the complete batch;
// parallelism is over (position, output-channel), not nested inference jobs.
void conv3_forward_batch(const std::vector<float>& in, int batch, int inC, int outC,
                         const std::vector<float>& w, const std::vector<float>& bias,
                         std::vector<float>& out, std::vector<float>& padded,
                         bool reluOut, std::stop_token stop) {
    out.assign(std::size_t(batch) * outC * kSquares, 0.0f);
    // Shared read-only by the OpenMP workers after construction. Capacity is
    // retained in BatchScratch, so steady-state layer execution does not
    // allocate this comparatively large padding tensor.
    padded.assign(std::size_t(batch) * inC * 100, 0.0f);
    for (int n = 0; n < batch; ++n)
        for (int ic = 0; ic < inC; ++ic) {
            const float* src = in.data() + (std::size_t(n) * inC + ic) * kSquares;
            float* dst = padded.data() + (std::size_t(n) * inC + ic) * 100;
            for (int r = 0; r < 8; ++r)
                std::memcpy(dst + (r + 1) * 10 + 1, src + r * 8, 8 * sizeof(float));
        }
#if defined(SCARLET_USE_OPENMP)
#pragma omp parallel for schedule(static) num_threads(inference_omp_threads())
#endif
    for (int task = 0; task < batch * outC; ++task) {
        if (stop.stop_requested()) continue;
        const int n = task / outC;
        const int oc = task % outC;
        float* dst = out.data() + (std::size_t(n) * outC + oc) * kSquares;
        std::fill_n(dst, kSquares, bias.empty() ? 0.0f : bias[oc]);
        const float* wb = w.data() + std::size_t(oc) * inC * 9;
        for (int ic = 0; ic < inC && !stop.stop_requested(); ++ic) {
            const float* src = padded.data() + (std::size_t(n) * inC + ic) * 100;
            const float* wk = wb + ic * 9;
            for (int r = 0; r < 8; ++r)
                for (int f = 0; f < 8; ++f) {
                    float sum = 0.0f;
                    int ki = 0;
                    for (int kr = 0; kr < 3; ++kr)
                        for (int kc = 0; kc < 3; ++kc, ++ki)
                            sum += wk[ki] * src[(r + kr) * 10 + f + kc];
                    dst[r * 8 + f] += sum;
                }
        }
        if (reluOut)
            for (int sq = 0; sq < kSquares; ++sq) dst[sq] = relu(dst[sq]);
    }
}

void conv1_forward_batch(const std::vector<float>& in, int batch, int inC, int outC,
                         const std::vector<float>& w, const std::vector<float>& bias,
                         std::vector<float>& out, bool reluOut, std::stop_token stop) {
    out.assign(std::size_t(batch) * outC * kSquares, 0.0f);
#if defined(SCARLET_USE_OPENMP)
#pragma omp parallel for schedule(static) num_threads(inference_omp_threads())
#endif
    for (int task = 0; task < batch * outC; ++task) {
        if (stop.stop_requested()) continue;
        const int n = task / outC;
        const int oc = task % outC;
        float* dst = out.data() + (std::size_t(n) * outC + oc) * kSquares;
        std::fill_n(dst, kSquares, bias.empty() ? 0.0f : bias[oc]);
        const float* ww = w.data() + std::size_t(oc) * inC;
        for (int ic = 0; ic < inC && !stop.stop_requested(); ++ic) {
            const float* src = in.data() + (std::size_t(n) * inC + ic) * kSquares;
            for (int sq = 0; sq < kSquares; ++sq) dst[sq] += ww[ic] * src[sq];
        }
        if (reluOut)
            for (int sq = 0; sq < kSquares; ++sq) dst[sq] = relu(dst[sq]);
    }
}

void fc_forward_batch(const std::vector<float>& in, int batch, int inSize,
                      const std::vector<float>& w, const std::vector<float>& bias,
                      int outSize, std::vector<float>& out, bool reluOut,
                      std::stop_token stop) {
    out.assign(std::size_t(batch) * outSize, 0.0f);
#if defined(SCARLET_USE_OPENMP)
#pragma omp parallel for schedule(static) num_threads(inference_omp_threads())
#endif
    for (int task = 0; task < batch * outSize; ++task) {
        if (stop.stop_requested()) continue;
        const int n = task / outSize;
        const int o = task % outSize;
        const float* src = in.data() + std::size_t(n) * inSize;
        const float* ww = w.data() + std::size_t(o) * inSize;
        float sum = bias.empty() ? 0.0f : bias[o];
        for (int i = 0; i < inSize && !stop.stop_requested(); ++i) sum += ww[i] * src[i];
        out[task] = reluOut ? relu(sum) : sum;
    }
}

std::array<float, 3> softmax3(const std::vector<float>& x) {
    std::array<float, 3> out{0.3333f, 0.3333f, 0.3333f};
    if (x.size() < 3) return out;
    const float m = std::max({x[0], x[1], x[2]});
    const float a = std::exp(x[0] - m), b = std::exp(x[1] - m), c = std::exp(x[2] - m);
    const float s = std::max(1e-20f, a + b + c);
    out = {a / s, b / s, c / s};
    return out;
}

Value wdl_to_cp(std::array<float, 3> wdl, Value fallbackCp) {
    // W - L is the expected result centred on zero. Keep the conversion
    // independent from Berserk: the old fallback-relative clamp made the
    // supposed second opinion inherit the primary evaluator's mistakes and
    // also made cached results depend on who happened to request them first.
    const double centered = std::clamp(double(wdl[0]) - double(wdl[2]), -0.995, 0.995);
    if (!std::isfinite(centered)) return fallbackCp;
    const double cp = 400.0 * std::atanh(centered);
    return clamp_cp(std::clamp(int(std::lround(cp)), -2200, 2200));
}

void set_plane(float* planes, int plane, int sq, float value = 1.0f) {
    if (plane >= 0 && plane < kInputPlanes && sq >= 0 && sq < 64)
        planes[std::size_t(plane) * kSquares + sq] = value;
}
void fill_plane(float* planes, int plane, float value = 1.0f) {
    if (plane < 0 || plane >= kInputPlanes) return;
    std::fill_n(planes + std::size_t(plane) * kSquares, kSquares, value);
}

int orient_sq(int sq, Color side) {
    // LC0 legacy encoding is side-to-move centric. Mirroring ranks for black keeps
    // "our pawns go up" for both sides without using LC0's full history object.
    return side == WHITE ? sq : (sq ^ 56);
}

Key raw_cache_key(const Position& pos) {
    Key h = pos.key ^ (0x9E3779B97F4A7C15ULL * Key(pos.halfmoveClock + 1));
    const int begin = std::max(0, int(pos.keyHistory.size()) - kMoveHistory);
    for (int i = begin; i < int(pos.keyHistory.size()); ++i) {
        const Key x = pos.keyHistory[i] + 0xBF58476D1CE4E5B9ULL + Key(i - begin + 1);
        h ^= (x << ((i - begin + 1) & 31)) | (x >> (64 - ((i - begin + 1) & 31)));
    }
    Key repetitionMask = 0;
    for (int i = 0; i < pos.leelaHistoryCount; ++i)
        if (pos.leelaHistory[i].repeated) repetitionMask |= Key(1) << i;
    h ^= repetitionMask * 0x94D049BB133111EBULL;
    return h;
}

int plane_piece_offset(PieceType pt) {
    switch (pt) {
        case PAWN: return 0;
        case KNIGHT: return 1;
        case BISHOP: return 2;
        case ROOK: return 3;
        case QUEEN: return 4;
        case KING: return 5;
        default: return -1;
    }
}

std::string default_weights_path() {
    const char* env = std::getenv("SCARLET_LD2_WEIGHTS");
    if (env && *env) return RuntimePaths::explicit_path(env).string();
    const std::array<const char*, 4> candidates{
        "assets/networks/policy_value.pb", "assets/networks/policy_value.pb.gz",
        "../assets/networks/policy_value.pb", "../assets/networks/policy_value.pb.gz"
    };
    for (const char* relative : candidates) {
        const auto path = RuntimePaths::default_asset(relative);
        if (std::filesystem::exists(path)) return path.string();
    }
    return RuntimePaths::default_asset("assets/networks/policy_value.pb").string();
}

} // namespace

RawBackend& RawBackend::instance() {
    static RawBackend b;
    return b;
}

bool RawBackend::try_load_default() {
    return load(default_weights_path());
}

bool RawBackend::load(std::string_view path) {
    clear_cache();
    net_ = Network{};
    const auto resolvedPath = RuntimePaths::explicit_path(path);
    status_.weights_path = resolvedPath.string();
    try {
        std::string bytes = read_weights_file(status_.weights_path);
        status_.weight_bytes = bytes.size();
        const auto top = parse_fields(bytes);
        bool magicOk = false;
        std::string_view weightsMsg, formatMsg, minVersionMsg;
        for (const auto& f : top) {
            if (f.id == 1 && f.wire == 5) magicOk = (f.fixed32 == kWeightMagic);
            else if (f.id == 2 && f.wire == 2) minVersionMsg = f.data;
            else if (f.id == 3 && f.wire == 2) formatMsg = f.data;
            else if (f.id == 10 && f.wire == 2) weightsMsg = f.data;
        }
        if (!magicOk) throw std::runtime_error("bad LC0 weights magic");
        if (formatMsg.empty()) throw std::runtime_error("missing network format");
        // LD2 is the legacy LINEAR16 residual/SE signature used by the bundled
        // network. Accepting another enum and merely guessing from tensor sizes
        // can produce numerically plausible but semantically invalid output.
        int encoding = -1, networkFormat = -1, inputFormat = -1;
        for (const auto& f : parse_fields(formatMsg)) {
            if (f.wire != 0) continue;
            if (f.id == 1) encoding = int(f.var);
            else if (f.id == 2) networkFormat = int(f.var);
            else if (f.id == 3) inputFormat = int(f.var);
        }
        if (encoding != 0) throw std::runtime_error("unsupported weights encoding (LINEAR16 required)");
        if (networkFormat != 21 || inputFormat != 0)
            throw std::runtime_error("unsupported LD2 network/input format");
        if (!minVersionMsg.empty()) {
            int major = 0, minor = 0, patch = 0;
            for (const auto& f : parse_fields(minVersionMsg)) {
                if (f.wire != 0) continue;
                if (f.id == 1) major = int(f.var);
                else if (f.id == 2) minor = int(f.var);
                else if (f.id == 3) patch = int(f.var);
            }
            if (std::array<int, 3>{major, minor, patch} > std::array<int, 3>{0, 32, 1})
                throw std::runtime_error("unsupported future LC0 weights version");
        }
        if (weightsMsg.empty()) throw std::runtime_error("missing weights message");

        auto residualMsgs = all_len_fields(weightsMsg, 2);
        std::string_view inputMsg, policyMsg, valueMsg, policy1Msg;
        std::string_view ip1wMsg, ip1bMsg, ip2wMsg, ip2bMsg;
        for (const auto& f : parse_fields(weightsMsg)) {
            if (f.wire != 2) continue;
            if (f.id == 1) inputMsg = f.data;
            else if (f.id == 3) policyMsg = f.data;
            else if (f.id == 6) valueMsg = f.data;
            else if (f.id == 7) ip1wMsg = f.data;
            else if (f.id == 8) ip1bMsg = f.data;
            else if (f.id == 9) ip2wMsg = f.data;
            else if (f.id == 10) ip2bMsg = f.data;
            else if (f.id == 11) policy1Msg = f.data;
        }

        net_.input = decode_convblock(inputMsg, kInputPlanes, 0, 3);
        net_.channels = net_.input.out_channels;
        if (net_.channels <= 0 || net_.channels > 1024)
            throw std::runtime_error("bad input channels");
        validate_conv(net_.input, "input convolution", kInputPlanes, net_.channels, 3);

        for (auto msg : residualMsgs) {
            ResidualBlock r;
            std::string_view c1 = first_len_field(msg, 1);
            std::string_view c2 = first_len_field(msg, 2);
            std::string_view se = first_len_field(msg, 3);
            r.conv1 = decode_convblock(c1, net_.channels, net_.channels, 3);
            r.conv2 = decode_convblock(c2, net_.channels, net_.channels, 3);
            if (!se.empty()) {
                r.se = decode_se(se);
                r.has_se = !r.se.w1.empty() && !r.se.w2.empty() && !r.se.b1.empty() && !r.se.b2.empty();
            }
            validate_conv(r.conv1, "residual conv1", net_.channels, net_.channels, 3);
            validate_conv(r.conv2, "residual conv2", net_.channels, net_.channels, 3);
            if (!se.empty() && !r.has_se)
                throw std::runtime_error("incomplete squeeze-excitation block");
            if (r.has_se) {
                const std::size_t hidden = r.se.b1.size();
                if (hidden == 0 || r.se.b2.size() != std::size_t(2 * net_.channels)
                    || r.se.w1.size() != hidden * std::size_t(net_.channels)
                    || r.se.w2.size() != std::size_t(2 * net_.channels) * hidden)
                    throw std::runtime_error("invalid squeeze-excitation dimensions");
            }
            net_.residual.push_back(std::move(r));
        }

        net_.policy1 = decode_convblock(policy1Msg, net_.channels, net_.channels, 3);
        net_.policy = decode_convblock(policyMsg, net_.channels, 0, 3);
        net_.policy_planes = net_.policy.out_channels;
        net_.value = decode_convblock(valueMsg, net_.channels, 0, 1);
        net_.value_planes = net_.value.out_channels;
        net_.ip1_val_w = decode_layer(ip1wMsg);
        net_.ip1_val_b = decode_layer(ip1bMsg);
        net_.ip2_val_w = decode_layer(ip2wMsg);
        net_.ip2_val_b = decode_layer(ip2bMsg);

        if ((net_.policy_planes != 73 && net_.policy_planes != 80) || net_.value_planes <= 0)
            throw std::runtime_error("bad head dimensions");
        if (net_.ip2_val_b.size() != 3)
            throw std::runtime_error("LD2 value head is not WDL/3-output");
        validate_conv(net_.policy1, "policy trunk", net_.channels, net_.channels, 3);
        validate_conv(net_.policy, "policy head", net_.channels, net_.policy_planes, 3);
        validate_conv(net_.value, "value head", net_.channels, net_.value_planes, 1);
        const std::size_t flatValue = std::size_t(net_.value_planes) * kSquares;
        const std::size_t valueHidden = net_.ip1_val_b.size();
        if (valueHidden == 0) throw std::runtime_error("empty value hidden layer");
        validate_fc("value fc1", net_.ip1_val_w, net_.ip1_val_b, flatValue, valueHidden);
        validate_fc("value fc2", net_.ip2_val_w, net_.ip2_val_b, valueHidden, 3);
        net_.valid = true;
        status_.loaded = true;
        status_.probes = 0;
        status_.cache_hits = 0;
        status_.cache_evictions = 0;
        status_.cache_entries = 0;
        status_.cache_capacity = cache_capacity_;
        status_.batch_calls = 0;
        status_.batch_positions = 0;
        status_.policy_queries = 0;
        status_.policy_hits = 0;
        status_.total_us = 0;
        status_.workspace_reallocations = 0;
        status_.channels = net_.channels;
        status_.residual_blocks = int(net_.residual.size());
        status_.policy_planes = net_.policy_planes;
        status_.value_planes = net_.value_planes;
#if defined(SCARLET_USE_OPENMP)
        status_.openmp = true;
        if (status_.threads <= 1) status_.threads = 4;
        omp_set_num_threads(status_.threads);
#else
        status_.openmp = false;
        status_.threads = 1;
#endif
        status_.message = "LD2 raw protobuf loaded: independent value+policy, 8-frame position history";
        reserve_batch_workspace(net_, 32);
        warm_kernel_team(batch_workers_requested_ > 0
            ? batch_workers_requested_ : std::max(1, status_.threads));
        return true;
    } catch (const std::exception& e) {
        status_.loaded = false;
        status_.message = std::string("LD2 raw load failed: ") + e.what();
        ++status_.failures;
        return false;
    }
}

void RawBackend::set_enabled(bool enabled) {
    status_.enabled = enabled;
}

void RawBackend::set_threads(int threads) {
    status_.threads = std::clamp(threads, 1, 16);
#if defined(SCARLET_USE_OPENMP)
    status_.openmp = true;
    omp_set_num_threads(status_.threads);
#else
    status_.threads = 1;
    status_.openmp = false;
    (void)threads;
#endif
    warm_kernel_team(status_.threads);
}

void RawBackend::set_batch_workers(int workers) {
    batch_workers_requested_ = std::clamp(workers, 0, 16);
    status_.batch_workers = batch_workers_requested_ > 0
        ? batch_workers_requested_ : std::max(1, status_.threads);
    warm_kernel_team(status_.batch_workers);
}

void RawBackend::set_cache_capacity(std::size_t entries) {
    cache_capacity_ = std::min<std::size_t>(entries, 262144);
    status_.cache_capacity = cache_capacity_;
    clear_cache();
}

void RawBackend::clear_cache() {
    cache_.clear();
    cache_lru_.clear();
    status_.cache_entries = 0;
}

void RawBackend::cache_result(Key key, RawProbeResult result) {
    if (cache_capacity_ == 0) return;
    if (auto existing = cache_.find(key); existing != cache_.end()) {
        existing->second.result = std::move(result);
        cache_lru_.splice(cache_lru_.end(), cache_lru_, existing->second.lru);
        existing->second.lru = std::prev(cache_lru_.end());
        return;
    }
    while (cache_.size() >= cache_capacity_ && !cache_lru_.empty()) {
        const Key victim = cache_lru_.front();
        cache_lru_.pop_front();
        cache_.erase(victim);
        ++status_.cache_evictions;
    }
    cache_lru_.push_back(key);
    cache_.emplace(key, CacheEntry{std::move(result), std::prev(cache_lru_.end())});
    status_.cache_entries = cache_.size();
}

std::vector<float> RawBackend::encode_planes(const Position& pos) const {
    std::vector<float> planes(std::size_t(kInputPlanes) * kSquares, 0.0f);
    encode_planes_into(pos, planes.data());
    return planes;
}

void RawBackend::encode_planes_into(const Position& pos, float* planes) const {
    std::fill_n(planes, std::size_t(kInputPlanes) * kSquares, 0.0f);
    const Color us = pos.side;
    const Color them = ~us;
    const int historyCount = std::min(pos.leelaHistoryCount, kMoveHistory);
    for (int history = 0; history < historyCount; ++history) {
        const LeelaHistoryFrame& frame = pos.leelaHistory[pos.leelaHistoryCount - 1 - history];
        const int base = history * kPlanesPerBoard;
        for (Color c : {WHITE, BLACK}) {
            for (PieceType pt : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
                Bitboard b = frame.pieces[int(c) * 6 + (int(pt) - int(PAWN))];
                while (b) {
                    const int sq = pop_lsb(b);
                    const int plane = base + (c == us ? 0 : 6) + plane_piece_offset(pt);
                    set_plane(planes, plane, orient_sq(sq, us));
                }
            }
        }
        if (frame.repeated) fill_plane(planes, base + 12, 1.0f);
    }

    const bool weQ = us == WHITE ? (pos.castlingRights & WQ) : (pos.castlingRights & BQ);
    const bool weK = us == WHITE ? (pos.castlingRights & WK) : (pos.castlingRights & BK);
    const bool thQ = them == WHITE ? (pos.castlingRights & WQ) : (pos.castlingRights & BQ);
    const bool thK = them == WHITE ? (pos.castlingRights & WK) : (pos.castlingRights & BK);
    if (weQ) fill_plane(planes, kAuxPlaneBase + 0, 1.0f);
    if (weK) fill_plane(planes, kAuxPlaneBase + 1, 1.0f);
    if (thQ) fill_plane(planes, kAuxPlaneBase + 2, 1.0f);
    if (thK) fill_plane(planes, kAuxPlaneBase + 3, 1.0f);
    if (us == BLACK) fill_plane(planes, kAuxPlaneBase + 4, 1.0f);
    fill_plane(planes, kAuxPlaneBase + 5, float(pos.halfmoveClock));
    fill_plane(planes, kAuxPlaneBase + 7, 1.0f);
}

int RawBackend::canonical_policy_plane(const Position& pos, Move move) {
    int from = orient_sq(from_sq(move), pos.side);
    int to = orient_sq(to_sq(move), pos.side);
    const int ff = file_of(from), fr = rank_of(from);
    const int tf = file_of(to), tr = rank_of(to);
    const int df = tf - ff;
    const int dr = tr - fr;

    // LC0 represents knight promotion through the ordinary queen-ray plane.
    // The nine dedicated planes are direction-major and encode R/B/Q.
    if (is_promotion(move) && promotion_type(move) != KNIGHT) {
        const int dir = df < 0 ? 0 : (df == 0 ? 1 : 2);
        int piece = -1;
        if (promotion_type(move) == ROOK) piece = 0;
        else if (promotion_type(move) == BISHOP) piece = 1;
        else if (promotion_type(move) == QUEEN) piece = 2;
        if (piece < 0) return -1;
        return 64 + dir * 3 + piece;
    }

    static constexpr std::array<std::pair<int,int>, 8> dirs{{
        {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}, {-1,0}, {-1,1}
    }};
    for (int d = 0; d < 8; ++d) {
        const int dx = dirs[d].first, dy = dirs[d].second;
        for (int dist = 1; dist <= 7; ++dist) {
            if (df == dx * dist && dr == dy * dist)
                return d * 7 + (dist - 1);
        }
    }
    static constexpr std::array<std::pair<int,int>, 8> knights{{
        {1,2}, {2,1}, {2,-1}, {1,-2}, {-1,-2}, {-2,-1}, {-2,1}, {-1,2}
    }};
    for (int k = 0; k < 8; ++k)
        if (df == knights[k].first && dr == knights[k].second) return 56 + k;
    return -1;
}

std::optional<RawProbeResult> RawBackend::run_network(
        const Position& pos, Value fallbackCp, std::stop_token stop) {
    if (!net_.valid || stop.stop_requested()) return std::nullopt;

    std::vector<float> b0 = encode_planes(pos), b1, b2, b3;
    conv3_forward(b0, kInputPlanes, net_.channels, net_.input.weights, net_.input.biases, b2, true);

    for (const auto& r : net_.residual) {
        if (stop.stop_requested()) return std::nullopt;
        conv3_forward(b2, net_.channels, net_.channels, r.conv1.weights, r.conv1.biases, b1, true);
        conv3_forward(b1, net_.channels, net_.channels, r.conv2.weights, {}, b3, false);
        if (r.has_se) {
            const int seOut = int(r.se.b1.size());
            std::vector<float> pool(net_.channels, 0.0f);
            for (int ch = 0; ch < net_.channels; ++ch) {
                float sum = 0.0f;
                for (int sq = 0; sq < kSquares; ++sq) sum += b3[std::size_t(ch) * kSquares + sq];
                pool[ch] = sum / float(kSquares) + r.conv2.biases[ch];
            }
            std::vector<float> fc1;
            fc_forward(pool, r.se.w1, r.se.b1, seOut, fc1, true);
            std::vector<float> scale;
            fc_forward(fc1, r.se.w2, r.se.b2, 2 * net_.channels, scale, false);
            std::vector<float> out(std::size_t(net_.channels) * kSquares);
            for (int ch = 0; ch < net_.channels; ++ch) {
                const float gamma = sigmoid(scale[ch]);
                const float beta = scale[ch + net_.channels] + gamma * r.conv2.biases[ch];
                for (int sq = 0; sq < kSquares; ++sq) {
                    const std::size_t idx = std::size_t(ch) * kSquares + sq;
                    out[idx] = relu(gamma * b3[idx] + b2[idx] + beta);
                }
            }
            b2.swap(out);
        } else {
            for (int ch = 0; ch < net_.channels; ++ch) {
                for (int sq = 0; sq < kSquares; ++sq) {
                    const std::size_t idx = std::size_t(ch) * kSquares + sq;
                    b3[idx] = relu(b3[idx] + r.conv2.biases[ch] + b2[idx]);
                }
            }
            b2.swap(b3);
        }
    }

    // Value head.
    if (stop.stop_requested()) return std::nullopt;
    conv1_forward(b2, net_.channels, net_.value_planes, net_.value.weights, net_.value.biases, b1, true);
    std::vector<float> flat = std::move(b1);
    std::vector<float> hidden;
    const int hiddenSize = int(net_.ip1_val_b.size());
    fc_forward(flat, net_.ip1_val_w, net_.ip1_val_b, hiddenSize, hidden, true);
    std::vector<float> logits;
    fc_forward(hidden, net_.ip2_val_w, net_.ip2_val_b, 3, logits, false);

    RawProbeResult result;
    result.valid = true;
    result.backend = "LD2/raw-protobuf/Scarlet-inprocess";
    result.wdl = softmax3(logits);
    result.cp = wdl_to_cp(result.wdl, fallbackCp);

    // Policy head. Used only for legal move priors; Scarlet still owns search/choice.
    conv3_forward(b2, net_.channels, net_.channels, net_.policy1.weights, net_.policy1.biases, b1, true);
    conv3_forward(b1, net_.channels, net_.policy_planes, net_.policy.weights, net_.policy.biases, b3, false);

    MoveList legal;
    Position tmp = pos;
    generate_legal(tmp, legal);
    result.policy.reserve(legal.size);
    float maxLogit = -std::numeric_limits<float>::infinity();
    std::vector<float> logitsLegal;
    logitsLegal.reserve(legal.size);
    for (Move m : legal) {
        const int plane = canonical_policy_plane(pos, m);
        float logit = -12.0f;
        if (plane >= 0 && plane < net_.policy_planes) {
            const int fsq = orient_sq(from_sq(m), pos.side);
            logit = b3[std::size_t(plane) * kSquares + fsq];
        }
        logitsLegal.push_back(logit);
        maxLogit = std::max(maxLogit, logit);
    }
    float denom = 0.0f;
    for (float x : logitsLegal) denom += std::exp(std::clamp(x - maxLogit, -80.0f, 80.0f));
    denom = std::max(denom, 1e-20f);
    for (int i = 0; i < legal.size; ++i) {
        PolicyEntry e;
        e.move = legal.moves[i];
        e.plane = canonical_policy_plane(pos, e.move);
        e.prior = std::exp(std::clamp(logitsLegal[i] - maxLogit, -80.0f, 80.0f)) / denom;
        result.policy.push_back(e);
    }
    std::sort(result.policy.begin(), result.policy.end(), [](const PolicyEntry& a, const PolicyEntry& b) {
        return a.prior > b.prior;
    });
    return result;
}

std::vector<std::optional<RawProbeResult>> RawBackend::run_network_batch(
        const std::vector<RawProbeRequest>& requests, std::stop_token stop) {
    const int batch = int(requests.size());
    std::vector<std::optional<RawProbeResult>> results(requests.size());
    if (!net_.valid || batch == 0 || stop.stop_requested()) return results;

    const int kernelThreads = batch_workers_requested_ > 0
        ? batch_workers_requested_ : std::max(1, status_.threads);
    const int oldThreads = tls_inference_threads;
    tls_inference_threads = std::clamp(kernelThreads, 1, 16);

    BatchScratch& scratch = persistent_batch_scratch;
    auto require_capacity = [&](std::vector<float>& workspace, std::size_t required) {
        if (workspace.capacity() >= required) return;
        ++status_.workspace_reallocations;
        workspace.reserve(required);
    };
    const std::size_t batchSize = std::size_t(batch);
    const int maxChannels = std::max({kInputPlanes, net_.channels, net_.policy_planes,
                                     net_.value_planes});
    require_capacity(scratch.b0, batchSize * kInputPlanes * kSquares);
    require_capacity(scratch.b1, batchSize * std::size_t(maxChannels) * kSquares);
    require_capacity(scratch.b2, batchSize * std::size_t(maxChannels) * kSquares);
    require_capacity(scratch.b3, batchSize * std::size_t(maxChannels) * kSquares);
    require_capacity(scratch.padded,
        batchSize * std::size_t(std::max(kInputPlanes, net_.channels)) * 100);
    require_capacity(scratch.pool, batchSize * std::size_t(net_.channels));
    require_capacity(scratch.hidden, batchSize * net_.ip1_val_b.size());
    require_capacity(scratch.scale, batchSize * std::size_t(2 * net_.channels));
    require_capacity(scratch.valueLogits, batchSize * 3);
    auto& b0 = scratch.b0;
    auto& b1 = scratch.b1;
    auto& b2 = scratch.b2;
    auto& b3 = scratch.b3;
    b0.resize(std::size_t(batch) * kInputPlanes * kSquares);
    for (int n = 0; n < batch && !stop.stop_requested(); ++n)
        encode_planes_into(requests[n].position,
            b0.data() + std::size_t(n) * kInputPlanes * kSquares);
    if (stop.stop_requested()) {
        tls_inference_threads = oldThreads;
        return results;
    }
    conv3_forward_batch(b0, batch, kInputPlanes, net_.channels,
                        net_.input.weights, net_.input.biases, b2, scratch.padded, true, stop);

    for (const auto& r : net_.residual) {
        if (stop.stop_requested()) {
            tls_inference_threads = oldThreads;
            return results;
        }
        conv3_forward_batch(b2, batch, net_.channels, net_.channels,
                            r.conv1.weights, r.conv1.biases, b1, scratch.padded, true, stop);
        if (stop.stop_requested()) {
            tls_inference_threads = oldThreads;
            return results;
        }
        conv3_forward_batch(b1, batch, net_.channels, net_.channels,
                            r.conv2.weights, {}, b3, scratch.padded, false, stop);
        if (stop.stop_requested()) {
            tls_inference_threads = oldThreads;
            return results;
        }
        if (r.has_se) {
            const int hidden = int(r.se.b1.size());
            auto& pool = scratch.pool;
            pool.assign(std::size_t(batch) * net_.channels, 0.0f);
            for (int n = 0; n < batch; ++n)
                for (int ch = 0; ch < net_.channels; ++ch) {
                    const float* src = b3.data() + (std::size_t(n) * net_.channels + ch) * kSquares;
                    float sum = 0.0f;
                    for (int sq = 0; sq < kSquares; ++sq) sum += src[sq];
                    pool[std::size_t(n) * net_.channels + ch]
                        = sum / float(kSquares) + r.conv2.biases[ch];
                }
            auto& fc1 = scratch.fc1;
            auto& scale = scratch.scale;
            fc_forward_batch(pool, batch, net_.channels, r.se.w1, r.se.b1,
                             hidden, fc1, true, stop);
            if (stop.stop_requested()) {
                tls_inference_threads = oldThreads;
                return results;
            }
            fc_forward_batch(fc1, batch, hidden, r.se.w2, r.se.b2,
                             2 * net_.channels, scale, false, stop);
            b1.assign(std::size_t(batch) * net_.channels * kSquares, 0.0f);
            for (int n = 0; n < batch; ++n)
                for (int ch = 0; ch < net_.channels; ++ch) {
                    const float gamma = sigmoid(scale[std::size_t(n) * 2 * net_.channels + ch]);
                    const float beta = scale[std::size_t(n) * 2 * net_.channels + ch + net_.channels]
                                     + gamma * r.conv2.biases[ch];
                    for (int sq = 0; sq < kSquares; ++sq) {
                        const std::size_t idx = (std::size_t(n) * net_.channels + ch) * kSquares + sq;
                        b1[idx] = relu(gamma * b3[idx] + b2[idx] + beta);
                    }
                }
            b2.swap(b1);
        } else {
            for (int n = 0; n < batch; ++n)
                for (int ch = 0; ch < net_.channels; ++ch)
                    for (int sq = 0; sq < kSquares; ++sq) {
                        const std::size_t idx = (std::size_t(n) * net_.channels + ch) * kSquares + sq;
                        b3[idx] = relu(b3[idx] + r.conv2.biases[ch] + b2[idx]);
                    }
            b2.swap(b3);
        }
    }

    if (stop.stop_requested()) {
        tls_inference_threads = oldThreads;
        return results;
    }
    conv1_forward_batch(b2, batch, net_.channels, net_.value_planes,
                        net_.value.weights, net_.value.biases, b1, true, stop);
    const int valueFlat = net_.value_planes * kSquares;
    auto& hidden = scratch.hidden;
    auto& valueLogits = scratch.valueLogits;
    const int hiddenSize = int(net_.ip1_val_b.size());
    fc_forward_batch(b1, batch, valueFlat, net_.ip1_val_w, net_.ip1_val_b,
                     hiddenSize, hidden, true, stop);
    fc_forward_batch(hidden, batch, hiddenSize, net_.ip2_val_w, net_.ip2_val_b,
                     3, valueLogits, false, stop);

    if (stop.stop_requested()) {
        tls_inference_threads = oldThreads;
        return results;
    }

    conv3_forward_batch(b2, batch, net_.channels, net_.channels,
                        net_.policy1.weights, net_.policy1.biases, b1, scratch.padded, true, stop);
    if (stop.stop_requested()) {
        tls_inference_threads = oldThreads;
        return results;
    }
    conv3_forward_batch(b1, batch, net_.channels, net_.policy_planes,
                        net_.policy.weights, net_.policy.biases, b3, scratch.padded, false, stop);

    for (int n = 0; n < batch; ++n) {
        if (stop.stop_requested()) break;
        RawProbeResult result;
        result.valid = true;
        result.backend = "LD2/raw-protobuf/Scarlet-batched";
        const std::vector<float> logits{
            valueLogits[std::size_t(n) * 3], valueLogits[std::size_t(n) * 3 + 1],
            valueLogits[std::size_t(n) * 3 + 2]};
        result.wdl = softmax3(logits);
        result.cp = wdl_to_cp(result.wdl, requests[n].fallback_cp);

        MoveList legal;
        Position tmp = requests[n].position;
        generate_legal(tmp, legal);
        std::vector<float> legalLogits;
        legalLogits.reserve(legal.size);
        float maxLogit = -std::numeric_limits<float>::infinity();
        for (Move move : legal) {
            const int plane = canonical_policy_plane(requests[n].position, move);
            float logit = -12.0f;
            if (plane >= 0 && plane < net_.policy_planes) {
                const int from = orient_sq(from_sq(move), requests[n].position.side);
                logit = b3[(std::size_t(n) * net_.policy_planes + plane) * kSquares + from];
            }
            legalLogits.push_back(logit);
            maxLogit = std::max(maxLogit, logit);
        }
        float denominator = 0.0f;
        for (float value : legalLogits)
            denominator += std::exp(std::clamp(value - maxLogit, -80.0f, 80.0f));
        denominator = std::max(denominator, 1e-20f);
        result.policy.reserve(legal.size);
        for (int i = 0; i < legal.size; ++i) {
            const Move move = legal.moves[i];
            result.policy.push_back(PolicyEntry{
                move,
                std::exp(std::clamp(legalLogits[i] - maxLogit, -80.0f, 80.0f)) / denominator,
                canonical_policy_plane(requests[n].position, move)});
        }
        std::sort(result.policy.begin(), result.policy.end(),
                  [](const PolicyEntry& a, const PolicyEntry& b) { return a.prior > b.prior; });
        results[n] = std::move(result);
    }
    tls_inference_threads = oldThreads;
    return results;
}

std::optional<RawProbeResult> RawBackend::probe(
        const Position& pos, Value fallbackCp, std::stop_token stop) {
    if (!status_.enabled || !status_.loaded || !net_.valid) return std::nullopt;
    const Key cacheKey = raw_cache_key(pos);
    auto it = cache_capacity_ ? cache_.find(cacheKey) : cache_.end();
    if (it != cache_.end()) {
        ++status_.cache_hits;
        cache_lru_.splice(cache_lru_.end(), cache_lru_, it->second.lru);
        it->second.lru = std::prev(cache_lru_.end());
        return it->second.result;
    }
    try {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<RawProbeRequest> request{{pos, fallbackCp}};
        auto batch = run_network_batch(request, stop);
        auto r = batch.empty() ? std::optional<RawProbeResult>{} : std::move(batch.front());
        const auto t1 = std::chrono::steady_clock::now();
        if (r && r->valid) {
            cache_result(cacheKey, *r);
            ++status_.probes;
            status_.total_us += std::uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
            return r;
        }
    } catch (const std::exception& e) {
        status_.message = std::string("LD2 raw inference failed: ") + e.what();
        ++status_.failures;
    }
    return std::nullopt;
}

std::optional<RawProbeResult> RawBackend::scalar_reference_probe(
        const Position& pos, Value fallbackCp) {
    if (!status_.enabled || !status_.loaded || !net_.valid) return std::nullopt;
    return run_network(pos, fallbackCp);
}

std::vector<std::optional<RawProbeResult>> RawBackend::probe_batch(
        const std::vector<RawProbeRequest>& requests, std::stop_token stop) {
    std::vector<std::optional<RawProbeResult>> results(requests.size());
    if (requests.empty() || !status_.enabled || !status_.loaded || !net_.valid)
        return results;

    struct Pending { std::size_t index; Key key; };
    std::vector<Pending> pending;
    std::vector<RawProbeRequest> uncached;
    pending.reserve(requests.size());
    uncached.reserve(requests.size());

    for (std::size_t i = 0; i < requests.size(); ++i) {
        const Key key = raw_cache_key(requests[i].position);
        auto it = cache_capacity_ ? cache_.find(key) : cache_.end();
        if (it != cache_.end()) {
            ++status_.cache_hits;
            cache_lru_.splice(cache_lru_.end(), cache_lru_, it->second.lru);
            it->second.lru = std::prev(cache_lru_.end());
            results[i] = it->second.result;
            continue;
        }
        pending.push_back(Pending{i, key});
        uncached.push_back(requests[i]);
    }
    if (pending.empty()) return results;

    const auto t0 = std::chrono::steady_clock::now();
    std::size_t completed = 0;
    try {
        auto batchResults = run_network_batch(uncached, stop);
        for (std::size_t i = 0; i < batchResults.size(); ++i) {
            if (!batchResults[i] || !batchResults[i]->valid || stop.stop_requested()) continue;
            results[pending[i].index] = std::move(batchResults[i]);
            cache_result(pending[i].key, *results[pending[i].index]);
            ++status_.probes;
            ++completed;
        }
    } catch (const std::exception& e) {
        status_.message = std::string("LD2 batch inference failed: ") + e.what();
        ++status_.failures;
    } catch (...) {
        status_.message = "LD2 batch inference failed: unknown exception";
        ++status_.failures;
    }
    const auto t1 = std::chrono::steady_clock::now();
    ++status_.batch_calls;
    status_.batch_positions += uncached.size();
    status_.batch_workers = batch_workers_requested_ > 0
        ? batch_workers_requested_ : std::max(1, status_.threads);
    // Accumulate batch wall time once: `total_us / probes` then represents the
    // effective per-position latency available to the search time manager.
    if (completed) {
        status_.total_us += std::uint64_t(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    }
    return results;
}

std::optional<RawProbeResult> RawBackend::cached_probe(const Position& pos) {
    if (!status_.enabled || !status_.loaded || !net_.valid) return std::nullopt;
    if (cache_capacity_ == 0) return std::nullopt;
    auto it = cache_.find(raw_cache_key(pos));
    if (it == cache_.end()) return std::nullopt;
    ++status_.cache_hits;
    cache_lru_.splice(cache_lru_.end(), cache_lru_, it->second.lru);
    it->second.lru = std::prev(cache_lru_.end());
    return it->second.result;
}

float RawBackend::raw_policy_prior_from_probe(const RawProbeResult& probe, Move move) const {
    const TTMove core = move_to_tt(move);
    for (const auto& e : probe.policy)
        if (same_move_core(e.move, core)) return e.prior;
    return 0.0f;
}

float RawBackend::cached_policy_prior(const Position& pos, Move move) {
    if (!status_.enabled || !status_.loaded || !net_.valid) return 0.0f;
    ++status_.policy_queries;
    Value fallback = VALUE_ZERO;
    if (auto p = probe(pos, fallback); p && p->valid) {
        const float prior = raw_policy_prior_from_probe(*p, move);
        if (prior > 0.0f) ++status_.policy_hits;
        return prior;
    }
    return 0.0f;
}

} // namespace Scarlet::Leela

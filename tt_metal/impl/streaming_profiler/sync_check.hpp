// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "impl/streaming_profiler/sync_engine.hpp"

namespace tt::tt_metal::streaming_profiler {

// Measures the synced timeline against an independent reference: the ruler's refclk readings on each chip, placed
// through the clock map, against the held-out link rounds solved onto the root. Logs the chip-to-chip error and each
// chip's AICLK over the capture at finish().
class SyncCheck {
public:
    struct Reading {
        double r;
        int64_t w8;
        float weight = 1;
    };
    struct Round {
        size_t li;
        RoundPoint p;
    };
    struct ChipBatch {
        std::vector<Reading> readings;
        std::optional<double> offset;
        std::vector<PlotPoint> aiclk;  // GHz, one point per instant on a line
    };
    struct Batch {
        explicit Batch(size_t devices = 0) : chips(devices) {}
        std::vector<Round> rounds;
        std::vector<ChipBatch> chips;
    };

    using PlotFn = std::function<void(const std::string&, std::span<const PlotPoint>)>;

    SyncCheck(const CaptureContext& ctx, const ClockMap& map);
    ~SyncCheck();
    SyncCheck(const SyncCheck&) = delete;
    SyncCheck& operator=(const SyncCheck&) = delete;

    // The batch the caller fills until submit().
    Batch& batch() { return staged_; }
    // Hands the batch to the check's thread and leaves it empty.
    void submit();
    // Stops the check's thread and logs the report.
    void finish();
    // The worst error per millisecond, pooled and per chip, as Tracy plot series.
    void plots(const PlotFn& plot) const;

private:
    struct ErrorStats {
        double n = 0.0, sum = 0.0, worst = 0.0;
        double worst_t = 0.0;
        void add(double ns, double t, double weight);
        double mean() const { return n > 0.0 ? sum / n : 0.0; }
    };
    struct ErrorHistogram {
        static constexpr int kBinsPerNs = 16;
        static constexpr int kRangeNs = 256;
        static constexpr int kCentre = kRangeNs * kBinsPerNs;
        ErrorStats stats;
        std::vector<double> bins = std::vector<double>(2 * kCentre, 0.0);
        void add(double ns, double t, double weight);
        double abs_quantile(double q) const;
    };
    // AICLK weighted by how long each value held.
    struct ClockStats {
        static constexpr double kBinMhz = 50.0;
        std::optional<PlotPoint> last;
        double span = 0.0, sum = 0.0, sum2 = 0.0;
        double lo = std::numeric_limits<double>::infinity(), hi = 0.0;
        uint64_t changes = 0;
        std::map<int, double> by_bin;
        void add(double t, double mhz);
    };
    struct LinkRef {
        std::deque<RoundPoint> rounds;
        std::map<int64_t, LineFit> lines;
        std::optional<int64_t> next;
    };
    struct Sample {
        double t;
        float err;
        float weight;
    };
    struct WorstByMs {
        int64_t first = 0;
        std::deque<float> ns;
        void add(double t, double e);
    };
    struct Chip {
        std::deque<Reading> waiting;
        std::optional<double> last_r;
        std::deque<Sample> placed;
        uint64_t popped = 0;
        uint64_t no_reference = 0, no_node = 0;
        std::optional<double> offset;
        ErrorStats err;
        WorstByMs worst;
        ClockStats clock;
        const Sample& at(uint64_t i) const { return placed[i - popped]; }
        uint64_t end() const { return popped + placed.size(); }
    };
    struct Pair {
        uint32_t a = 0, b = 0;
        uint64_t ia = 0, jb = 0;
        ErrorStats err;
    };
    enum class Ref { kReady, kWait, kNone };
    // A step's line needs its whole span of rounds, except once the capture is finishing.
    enum class Steps { Complete, All };
    using Mesh = std::vector<std::optional<RootXf>>;

    int64_t sender_step(size_t li, double root) const;
    double root_step(uint32_t dev, double r) const;
    static void fit_steps(LinkRef& ref, Steps steps);
    void absorb(const Round& in);
    Ref mesh_at(int64_t k, const Mesh*& out);
    Ref reference(uint32_t dev, double r, double& root);
    bool place(uint32_t dev, ClockMap::Reader& reader);
    bool pair_up(double until);
    void prune();
    void run();
    void stop();
    void report() const;

    const CaptureContext ctx_;
    const ClockMap& map_;
    std::vector<LinkRef> links_;
    std::vector<Chip> chips_;
    std::vector<Pair> pairs_;
    std::map<int64_t, Mesh> meshes_;
    ErrorHistogram pooled_;
    WorstByMs worst_;
    bool finishing_seen_ = false;
    Batch staged_;
    std::mutex mu_;
    Batch in_;
    bool finishing_ = false;
    std::thread worker_;
};

}  // namespace tt::tt_metal::streaming_profiler

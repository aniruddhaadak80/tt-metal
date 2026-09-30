// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

#include "hostdev/streaming_profiler_sync.h"
#include "impl/streaming_profiler/clock_map.hpp"
#include "impl/streaming_profiler/capture_context.hpp"

namespace tt::tt_metal::streaming_profiler {

// The x mean accumulates in long double, since x may be a count far from its spread.
struct LineFit {
    double x_mean = 0.0, y_mean = 0.0, slope = 0.0;
};
template <
    std::ranges::forward_range Points,
    std::invocable<std::ranges::range_reference_t<const Points>> X,
    std::invocable<std::ranges::range_reference_t<const Points>> Y>
LineFit fit_line(const Points& pts, X x, Y y) {
    long double sx = 0;
    double sy = 0.0;
    size_t n = 0;
    for (const auto& p : pts) {
        sx += std::invoke(x, p);
        sy += std::invoke(y, p);
        n++;
    }
    LineFit f{.x_mean = static_cast<double>(sx / static_cast<long double>(n)), .y_mean = sy / static_cast<double>(n)};
    double sxx = 0.0, sxy = 0.0;
    for (const auto& p : pts) {
        const double dx = std::invoke(x, p) - f.x_mean;
        sxx += dx * dx;
        sxy += dx * (std::invoke(y, p) - f.y_mean);
    }
    f.slope = sxy / sxx;
    return f;
}

// Edge i says x[ends[i][0]] - x[ends[i][1]] = value[i], an empty end being the ground, which is fixed at 0.
struct Potential {
    std::vector<double> x;
    std::optional<size_t> unreached;
};
Potential solve_potential(
    std::span<const std::array<std::optional<size_t>, 2>> ends, std::span<const double> value, size_t unknowns);

static_assert(1'000'000'000 % kernel_profiler::kEthRefclkHz == 0);
inline constexpr int64_t kNsPerRefclk = 1'000'000'000 / kernel_profiler::kEthRefclkHz;

// One link round in the sender's refclk ticks: the midpoint of its stamps, and the receiver's offset from it there.
struct RoundPoint {
    double mid = 0.0, off = 0.0;
};

// A chip's refclk onto the root's.
struct RootXf {
    double scale = 1.0, shift = 0.0;
    double operator()(double refclk) const { return scale * refclk + shift; }
};

// A point of a Tracy plot, at a root refclk tick.
struct PlotPoint {
    double root = 0.0, value = 0.0;
};

template <std::predicate<size_t> Usable>
std::vector<bool> reached_from_root(std::span<const CaptureContext::Link> links, size_t devices, Usable usable) {
    std::vector<bool> reached(devices, false);
    reached[0] = true;
    for (bool grew = true; grew;) {
        grew = false;
        for (size_t li = 0; li < links.size(); li++) {
            const CaptureContext::Link& L = links[li];
            if (reached[L.dev_a] != reached[L.dev_b] && usable(li)) {
                reached[L.dev_a] = reached[L.dev_b] = true;
                grew = true;
            }
        }
    }
    return reached;
}

// Chains each link's line (null for a link with none) onto the root chip; a chip no line reaches has none.
std::vector<std::optional<RootXf>> compose_on_root(const CaptureContext& ctx, std::span<const LineFit* const> lines);

class SyncCheck;

// LINK samples feed the link solver, refclk against refclk, so DVFS on either wall clock cannot enter the link solve.
class SyncEngine {
public:
    // Starts a capture on `ctx`, whose every device has a path over its links to the root (device index 0), publishing
    // into `map`.
    SyncEngine(const CaptureContext& ctx, ClockMap& map);
    ~SyncEngine();
    SyncEngine(const SyncEngine&) = delete;
    SyncEngine& operator=(const SyncEngine&) = delete;

    void on_record(uint32_t dev, uint32_t core, const kernel_profiler::SyncRecord& rec);
    // Publishes what the batch's records added to the map, and returns whether anything was.
    bool on_batch_end();
    void on_capture_end();

private:
    struct Instant {
        int64_t r = 0;
        int64_t w8 = 0;
        uint32_t k8 = 0;
        double wall() const { return static_cast<double>(w8) / 8.0; }
        int64_t wall_tick() const { return (w8 + 4) >> 3; }
    };
    // Values are offsets from the chip's bases, so no double holds a count since power-on.
    struct Chip {
        std::optional<int64_t> refclk_base, wall_base;
        std::deque<Instant> pts;
        size_t published = 0;
        const char* aiclk_plot = nullptr;
    };
    struct Round {
        std::array<std::optional<double>, 4> t;
        std::optional<double>& operator[](kernel_profiler::SyncRole role) { return t[static_cast<size_t>(role)]; }
        bool complete() const {
            return std::ranges::all_of(t, [](const auto& v) { return v.has_value(); });
        }
    };
    struct Link {
        std::map<uint32_t, Round> pending;
        std::vector<RoundPoint> rounds;
        std::optional<LineFit> line;
        double solved_at = -std::numeric_limits<double>::infinity();
    };
    struct CoreRef {
        uint32_t dev = 0, core = 0;
        auto operator<=>(const CoreRef&) const = default;
    };
    struct LinkSide {
        size_t link = 0;
        bool sender = false;
    };
    // A solve waits for a full window of rounds, except at capture end, which takes whatever a link has.
    enum class Window { Full, Partial };

    void on_stamp(uint32_t dev, uint32_t core, const kernel_profiler::SyncLinkRecord& s);
    void solve(Link& link, Window window);
    const std::vector<std::optional<RootXf>>& transforms();
    bool publish_dev(uint32_t dev);
    void plot(const char* name, std::span<const PlotPoint> series);

    ClockMap& map_;
    ClockMap::Reader reader_;
    CaptureContext ctx_;
    std::vector<Chip> chips_;
    std::vector<Link> links_;
    std::map<CoreRef, LinkSide> side_of_;
    std::vector<std::optional<RootXf>> to_root_;
    bool links_moved_ = true;
    std::vector<PlotPoint> aiclk_;
    std::unique_ptr<SyncCheck> check_;
};

}  // namespace tt::tt_metal::streaming_profiler

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// AICLK only takes whole FBDIV values, each a multiple of the crystal the refclk counts, so wherever it holds, the wall
// clock gains exactly k8/8 ticks per refclk tick, k8 = FBDIV * 8 / (REFDIV * postdiv0), and its samples lie on one
// line of that slope to within a sample's width.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_sync.h"
#include "tt_metal/impl/streaming_profiler/kernels/eth_sync_ring.hpp"

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl_addr");
constexpr uint32_t kSyncRingAddr = get_named_compile_time_arg_val("sync_ring_addr");
constexpr uint32_t kSampleRingAddr = get_named_compile_time_arg_val("sample_ring_addr");

namespace kp = kernel_profiler;

// A window becomes a point at least once a millisecond, well within a SyncLocalStep's 16-bit refclk field.
constexpr uint32_t kPointTicks = kp::kEthRefclkHz / 1000;
// A sample is placed to within half its read pair, at most 1.5 cycles, so a sample twice that far from the line is off
// it.
constexpr int32_t kOffEighths = 24;
// The sampler sees nearly every refclk update, one per 80 ns (4 ticks). A step off the line adds an eighth of a tick
// per update, so after eight samples (32 ticks) it has left the line. A sample only joins a window once kHold samples
// after it are on the line too.
constexpr uint32_t kHold = 8;
// Off a line, every kGroup consecutive samples (~320 ns, over which a clock walk bends the wall ~0.06 cycles) become
// one point, their centroid. A new line opens once kSteady such points (~4 us) fit one k8; a PLL step takes ~1.3 us, so
// it can't fit a line in that span.
constexpr uint32_t kGroup = 4;
constexpr uint32_t kSteady = 12;

using Out = SyncRingWriter<kCtrlAddr, kSyncRingAddr>;
constexpr kp::SyncMeta kMetaLocal{.kind = kp::SyncKind::Local};

struct Anchor {
    uint64_t R = 0, W = 0;
    uint32_t r = 0, w = 0;
    uint64_t full_r(uint32_t lo) const {
        return R + static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(lo - r)));
    }
    uint64_t full_w(uint32_t lo) const {
        return W + static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(lo - w)));
    }
    void move(uint32_t lr, uint32_t lw) {
        R = full_r(lr);
        W = full_w(lw);
        r = lr;
        w = lw;
    }
};

using Sample = kp::SyncSample;

struct Model {
    // The line the samples are on (k8 0 when there's none): through `at` at k8 eighths of a tick per refclk tick, with
    // the mean residue of its n samples so far.
    struct Line {
        uint32_t k8 = 0;
        Sample at{};
        int32_t mean = 0;
        int64_t sum = 0;
        uint32_t n = 0;
    };
    // The samples on the line since the last point. sum_res is their residues minus the line's mean; only a window
    // close moves the mean, so it fits in 32 bits.
    struct Window {
        Sample first{};
        uint32_t sum_dr = 0;
        int32_t sum_res = 0;
        uint32_t n = 0, size = 1;
        uint32_t last_r = 0;
    };
    // A held sample's residue is recomputed when it joins the window, since a line change empties the hold.
    struct Hold {
        Sample s[kHold];
        uint32_t beg = 0, n = 0;
    };
    // Off a line: the samples since the last point, and the last kSteady points.
    struct Group {
        Sample first{}, last{};
        uint32_t sum_dr = 0, sum_dw = 0, n = 0;
    };
    struct Recent {
        Sample s[kSteady];
        uint32_t n = 0, head = 0;
    };

    Line line;
    uint32_t slope = 0;
    Window win;
    Hold hold;
    Sample off{};
    bool has_off = false;
    Group group;
    Recent recent;
    Anchor anchor;
    Out out;

    FORCE_INLINE int32_t residue(Sample x) const {
        return static_cast<int32_t>((x.wall8 - line.at.wall8) - line.k8 * (x.refclk - line.at.refclk));
    }
    void point(uint32_t r, uint32_t w, uint32_t k) {
        out.add(anchor.full_r(r), anchor.full_w(w), k, slope, kMetaLocal);
    }
    __attribute__((noinline)) void close_window() {
        const uint32_t n = win.n;
        const uint32_t dr = (win.sum_dr + n / 2u) / n;
        const int32_t half = static_cast<int32_t>(n / 2u);
        // The window's error against its first sample: the sum of each sample's residue minus the first's.
        const int32_t se = win.sum_res - static_cast<int32_t>(n) * (residue(win.first) - line.mean);
        const int32_t e = (se + (se < 0 ? -half : half)) / static_cast<int32_t>(n);
        anchor.move(win.first.refclk, win.first.wall8);
        point(win.first.refclk + dr, win.first.wall8 + line.k8 * dr + static_cast<uint32_t>(e), line.k8);
        line.sum += static_cast<int64_t>(win.sum_res) + static_cast<int64_t>(n) * line.mean;
        line.n += n;
        line.mean = static_cast<int32_t>(line.sum / static_cast<int64_t>(line.n));
        win.n = 0;
        win.size = win.size < (1u << 23) ? win.size * 2u : win.size;
    }
    FORCE_INLINE void window_add(Sample x, int32_t res) {
        if (win.n == 0) {
            win.first = x;
            win.sum_dr = 0;
            win.sum_res = 0;
        }
        const uint32_t dr = x.refclk - win.first.refclk;
        win.sum_dr += dr;
        win.sum_res += res - line.mean;
        win.last_r = x.refclk;
        if (++win.n == win.size || dr >= kPointTicks) {
            close_window();
        }
    }
    __attribute__((noinline)) void close_group() {
        Group& g = group;
        const uint32_t span = g.last.refclk - g.first.refclk;
        if (span != 0) {
            slope = ((g.last.wall8 - g.first.wall8) + span / 2u) / span;
        }
        // Every group but the capture's last closes full, so the division is nearly always by the constant.
        const auto div = [&](uint32_t x) { return g.n == kGroup ? x / kGroup : x / g.n; };
        const uint32_t dr = div(g.sum_dr), q = g.sum_dr - dr * g.n;
        const uint32_t dw = div(g.sum_dw - slope * q + g.n / 2u);
        anchor.move(g.first.refclk, g.first.wall8);
        const Sample c{g.first.refclk + dr, g.first.wall8 + dw};
        point(c.refclk, c.wall8, 0);
        recent.s[(recent.head + recent.n) % kSteady] = c;
        if (recent.n < kSteady) {
            recent.n++;
        } else {
            recent.head = (recent.head + 1) % kSteady;
        }
        g.n = 0;
        if (recent.n == kSteady) {
            try_line();
        }
    }
    FORCE_INLINE void group_add(Sample x) {
        if (group.n == 0) {
            group.first = x;
            group.sum_dr = 0;
            group.sum_dw = 0;
        }
        group.sum_dr += x.refclk - group.first.refclk;
        group.sum_dw += x.wall8 - group.first.wall8;
        group.last = x;
        if (++group.n == kGroup) {
            close_group();
        }
    }
    __attribute__((noinline)) void try_line() {
        const Sample a = recent.s[recent.head];
        const Sample& z = recent.s[(recent.head + kSteady - 1) % kSteady];
        const uint32_t dr = z.refclk - a.refclk;
        if (dr == 0) {
            return;
        }
        const uint32_t k = ((z.wall8 - a.wall8) + dr / 2u) / dr;
        int32_t res[kSteady];
        int32_t sum = 0;
        for (uint32_t i = 0; i < kSteady; i++) {
            const Sample& x = recent.s[(recent.head + i) % kSteady];
            res[i] = static_cast<int32_t>((x.wall8 - a.wall8) - k * (x.refclk - a.refclk));
            sum += res[i];
        }
        const int32_t m = sum / static_cast<int32_t>(kSteady);
        for (uint32_t i = 0; i < kSteady; i++) {
            if (static_cast<uint32_t>(res[i] - m + kOffEighths) > 2u * kOffEighths) {
                return;
            }
        }
        line.k8 = k;
        line.at = a;
        line.mean = m;
        line.sum = sum;
        line.n = kSteady;
        slope = k;
        win.n = 0;
        win.size = 1;
        hold.n = 0;
        has_off = false;
    }
    // End the line with a point where its samples last held it. Otherwise the chord from the last window's centroid, up
    // to half a window back, into the first group would leave the line well before the clock did.
    __attribute__((noinline)) void end_line(Sample x) {
        if (win.n != 0) {
            close_window();
        }
        if (line.n > kSteady) {
            const uint32_t r = win.last_r;
            point(r, line.at.wall8 + line.k8 * (r - line.at.refclk) + static_cast<uint32_t>(line.mean), line.k8);
        }
        line.k8 = 0;
        recent.n = 0;
        recent.head = 0;
        group.n = 0;
        for (uint32_t i = 0; i < hold.n; i++) {
            group_add(hold.s[(hold.beg + i) % kHold]);
        }
        hold.n = 0;
        group_add(off);
        group_add(x);
        has_off = false;
    }
    FORCE_INLINE void on_sample(Sample x) {
        if (line.k8 == 0) {
            group_add(x);
            return;
        }
        const int32_t res = residue(x);
        if (static_cast<uint32_t>(res - line.mean + kOffEighths) > 2u * kOffEighths) {
            if (has_off) {
                end_line(x);
                return;
            }
            off = x;
            has_off = true;
            return;
        }
        has_off = false;
        if (hold.n == kHold) {
            window_add(hold.s[hold.beg], residue(hold.s[hold.beg]));
            hold.beg = (hold.beg + 1) % kHold;
            hold.n--;
        }
        hold.s[(hold.beg + hold.n) % kHold] = x;
        hold.n++;
    }
    // Feeds the samples [p, end) in ring order. On a line, with a full hold and nothing pending off it, the state a
    // sample touches stays in registers; anything else goes through on_sample.
    void feed(const volatile tt_l1_ptr Sample* p, const volatile tt_l1_ptr Sample* end) {
        const auto load = [](const volatile tt_l1_ptr Sample* q) { return Sample{q->refclk, q->wall8}; };
        while (p != end) {
            if (line.k8 == 0 || has_off || hold.n != kHold) {
                on_sample(load(p));
                p++;
                continue;
            }
            const uint32_t kk = line.k8, llr = line.at.refclk, llw = line.at.wall8;
            int32_t mn = line.mean, wsum = win.sum_res;
            uint32_t rr0 = win.first.refclk, ssr = win.sum_dr, n = win.n, sz = win.size, hb = hold.beg,
                     lr_last = win.last_r;
            bool left = false;
            // Load each sample's words and hold slot an iteration ahead, so their latency hides behind the math. The
            // one read past the end lands in L1 and is never used.
            Sample x = load(p), h = hold.s[hb];
            for (; p != end; p++) {
                const Sample xn = load(p + 1);
                const uint32_t hbn = (hb + 1) % kHold;
                const Sample hn = hold.s[hbn];
                const int32_t res = static_cast<int32_t>((x.wall8 - llw) - kk * (x.refclk - llr));
                if (static_cast<uint32_t>(res - mn + kOffEighths) > 2u * kOffEighths) {
                    left = true;
                    break;
                }
                hold.s[hb] = x;
                hb = hbn;
                if (n == 0) {
                    rr0 = h.refclk;
                    win.first.wall8 = h.wall8;
                    ssr = 0;
                    wsum = 0;
                }
                const uint32_t dr = h.refclk - rr0;
                ssr += dr;
                wsum += static_cast<int32_t>((h.wall8 - llw) - kk * (h.refclk - llr)) - mn;
                lr_last = h.refclk;
                if (++n == sz || dr >= kPointTicks) {
                    win.first.refclk = rr0;
                    win.sum_dr = ssr;
                    win.sum_res = wsum;
                    win.n = n;
                    win.last_r = lr_last;
                    close_window();
                    n = win.n;
                    sz = win.size;
                    mn = line.mean;
                }
                x = xn;
                h = hn;
            }
            win.first.refclk = rr0;
            win.sum_dr = ssr;
            win.sum_res = wsum;
            win.n = n;
            hold.beg = hb;
            win.last_r = lr_last;
            if (left) {
                on_sample(load(p));
                p++;
            }
        }
    }
    void finish() {
        if (line.k8 != 0) {
            for (uint32_t i = 0; i < hold.n; i++) {
                const Sample& x = hold.s[(hold.beg + i) % kHold];
                window_add(x, residue(x));
            }
            hold.n = 0;
            if (win.n != 0) {
                close_window();
            }
        } else if (group.n != 0) {
            close_group();
        }
    }
};

void kernel_main() {
    volatile tt_l1_ptr kp::RelayCtrl* ctl = Out::ctrl();
    volatile tt_l1_ptr kp::SyncSampleRing* ring =
        reinterpret_cast<volatile tt_l1_ptr kp::SyncSampleRing*>(kSampleRingAddr);
    Model m;
    uint32_t next = 0;
    do {
        invalidate_l1_cache();
    } while (ring->tail == 0 && ring->done == 0);
    invalidate_l1_cache();
    const uint64_t r = ring->refclk, w = ring->wall8;
    m.anchor = Anchor{.R = r, .W = w, .r = static_cast<uint32_t>(r), .w = static_cast<uint32_t>(w)};
    constexpr uint32_t kChunk = 64, kRing = kp::kSyncSampleRingSamples;
    uint32_t stop = 0;
    while (true) {
        invalidate_l1_cache();
        const bool done = ring->done != 0;
        stop = ctl->stop != 0u ? kp::kSyncHeadStop : stop;
        const uint32_t tail = ring->tail;
        const uint32_t end = next + std::min(tail - next, kChunk);
        const uint32_t a = next % kRing;
        const uint32_t first = std::min(end - next, kRing - a);
        m.feed(&ring->samples[a], &ring->samples[a + first]);
        m.feed(&ring->samples[0], &ring->samples[end - next - first]);
        next = end;
        std::atomic_thread_fence(std::memory_order_release);
        ring->head = next + stop;
        if (done && next == tail) {
            break;
        }
    }
    m.finish();
    m.out.close();
}

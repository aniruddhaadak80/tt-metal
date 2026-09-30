// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(ARCH_BLACKHOLE)
#error "eth_clock_stream.hpp is Blackhole only"
#endif

#include <cstdint>

#include "tt_metal/impl/streaming_profiler/kernels/eth_clock_sampling.hpp"

namespace sampler {
struct StreamArgs {
    volatile uint32_t* refclk;
    volatile uint32_t* wall;
    uint32_t period, pos8s;
    volatile uint32_t* slots;
    uint32_t mask;
    volatile uint32_t* head;
    uint32_t limit, iters;
    const uint32_t* pads;
    volatile uint32_t* tail_word;
    uint32_t tail;
};

struct Block {
    uint32_t r1, r2, w, r3;
};
enum class Step : uint8_t { Stored, Refill, Rejected };

/**
 * @brief Streams refclk updates into args->slots from args->tail on and returns the tail after the last one it stored.
 *
 * Update n goes to slot n & args->mask. Each pass first runs args->pads[tail & 255] / 4 nops, then reads blocks of
 * refclk, refclk, wall, refclk until the refclk steps. A update whose two blocks each took args->period wall ticks
 * is stored as its new refclk and the wall time of its step in eighths, the block's wall read plus its gap's signed
 * byte in args->pos8s; one that did not is dropped. With period 0 every update is stored, unpublished, as its block's
 * wall length, for calibration. Otherwise, before each stored update the tail it follows is written to
 * *args->tail_word, so every published sample was stored a handler earlier.
 *
 * At args->limit the stream reloads *args->head, the model's position, and moves the limit args->mask past it,
 * waiting there while the ring is full; the pass after a reload skips its pad, which pays for the reload. It returns
 * when the head is kSyncHeadStop past the model's position, or once args->iters passes have stored nothing (0 counts
 * 2^32).
 */
// -Os lays the calibration path in line and merges the handlers' tails, and the extra taken branches, which cost more
// with the branch predictor off, made the stream miss every other update at 800 MHz.
__attribute__((noinline, aligned(64), optimize("no-crossjumping", "reorder-blocks-algorithm=stc"))) inline uint32_t
sampler_stream(StreamArgs* args) {
    volatile uint32_t* const refclk = args->refclk;
    volatile uint32_t* const wall = args->wall;
    const uint32_t period = args->period, pos8s = args->pos8s, mask = args->mask;
    volatile uint32_t* const slots = args->slots;
    volatile uint32_t* const head = args->head;
    uint32_t limit = args->limit, iters = args->iters;
    const uint32_t* const pads = args->pads;
    volatile uint32_t* const tail_word = args->tail_word;
    uint32_t tail = args->tail;
    const auto read = [&]() __attribute__((always_inline)) {
        Block b;
        b.r1 = *refclk;
        b.r2 = *refclk;
        b.w = *wall;
        b.r3 = *refclk;
        return b;
    };
    // The pad jumps pad bytes back from the end of its nops. The asm defines the label, so it must be emitted once.
    uint32_t nops_end;
    asm("lla %0, .Lsampler_stream_nops_end" : "=r"(nops_end));
    uint32_t pad = 0;
    branch_predictor(false);
    // The pass starts on a cache line, so its blocks sit where they do in every build. Passing pad through keeps its
    // load ahead of the alignment, so nothing lands between the alignment and the pass.
    asm volatile(".p2align 6" : "+r"(pad));
    // Odds this low keep a step at the last check a single taken branch into its handler.
    const auto stepped = [](const Block& before, const Block& last) __attribute__((always_inline)) {
        return __builtin_expect_with_probability(last.r3 != before.r3, 0, 0.001);
    };
    // A step between before's and last's final reads is placed by last's reads, and now times the block after it.
    const auto store = [&](const Block& before, const Block& last, const Block& now)
                           __attribute__((always_inline)) -> Step {
        const uint32_t d0 = last.w - before.w, d1 = now.w - last.w;
        volatile uint32_t* const slot = slots + 2 * (tail & mask);
        if (__builtin_expect(period != 0, 1)) {
            if (((d0 ^ period) | (d1 ^ period)) != 0) {
                return Step::Rejected;
            }
            const uint32_t gap = static_cast<uint32_t>(last.r1 == before.r3) << (last.r2 == last.r1);
            slot[0] = last.r3;
            slot[1] = (last.w << 3) + static_cast<uint32_t>(static_cast<int8_t>(pos8s >> (8 * gap)));
            *tail_word = tail;
            return ++tail == limit ? Step::Refill : Step::Stored;
        }
        slot[0] = d1;
        return ++tail == limit ? Step::Refill : Step::Stored;
    };
    // Reads blocks until the refclk steps, and stores the step. Three blocks rotating by name keep each check's reads
    // in the registers they landed in; a single rotating site or an array makes the compiler copy them into one
    // handler's registers.
    const auto run = [&]() __attribute__((always_inline)) -> Step {
        Block a = read(), b = read(), c;
#pragma GCC unroll 6
        for (uint32_t i = 0; i < 6; i++) {
            c = read();
            if (stepped(a, b)) {
                return store(a, b, c);
            }
            a = read();
            if (stepped(b, c)) {
                return store(b, c, a);
            }
            b = read();
            if (stepped(c, a)) {
                return store(c, a, b);
            }
        }
        return Step::Rejected;
    };
pass:
    asm volatile(
        ".option push\n\t.option norvc\n\t"
        "sub t0, %[end], %[pad]\n\t"
        "jr t0\n\t"
        ".rept %[n]\n\tnop\n\t.endr\n"
        ".Lsampler_stream_nops_end:\n\t"
        ".option pop"
        :
        : [end] "r"(nops_end), [pad] "r"(pad), [n] "i"(kTrackerPadRange - 1)
        : "t0", "memory");
    pad = pads[tail & 255];
    switch (run()) {
        case Step::Stored: goto pass;
        case Step::Refill: goto refill;
        case Step::Rejected: goto reject;
    }
reject:
    if (--iters != 0) {
        goto pass;
    }
    goto done;
refill:
    for (;;) {
        const uint32_t h = *head;
        if (__builtin_expect(static_cast<int32_t>(tail - h) < 0, 0)) {
            goto done;
        }
        limit = h + mask;
        if (limit != tail) {
            break;
        }
    }
    pad = 0;
    goto pass;
done:
    branch_predictor(true);
    return tail;
}
}  // namespace sampler

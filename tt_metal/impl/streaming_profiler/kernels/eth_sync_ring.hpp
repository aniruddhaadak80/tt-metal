// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_sync.h"

template <uint32_t CtrlAddr, uint32_t RingAddr>
struct SyncRingWriter {
    uint32_t tail = 0, dropped = 0;
    uint32_t n = 0, steps[kernel_profiler::kSyncLocalPoints - 1] = {};
    kernel_profiler::SyncMeta meta{};
    kernel_profiler::SyncLocalRates rates{};
    uint64_t r0 = 0, w0 = 0;

    static volatile tt_l1_ptr kernel_profiler::RelayCtrl* ctrl() {
        return reinterpret_cast<volatile tt_l1_ptr kernel_profiler::RelayCtrl*>(CtrlAddr);
    }
    void emit() {
        invalidate_l1_cache();
        if (tail - ctrl()->sync_head >= kernel_profiler::kSyncRingRecords) {
            dropped++;
            return;
        }
        volatile tt_l1_ptr kernel_profiler::SyncLocalRecord* r =
            &reinterpret_cast<volatile tt_l1_ptr kernel_profiler::SyncRecord*>(
                 RingAddr)[tail % kernel_profiler::kSyncRingRecords]
                 .local;
        kernel_profiler::SyncMeta m = meta;
        m.count = n;
        r->meta = kernel_profiler::word_of(m);
        r->rates = kernel_profiler::word_of(rates);
        r->refclk = r0;
        r->wall8 = w0;
        r->steps[0] = steps[0];
        r->steps[1] = steps[1];
        std::atomic_thread_fence(std::memory_order_release);
        ctrl()->sync_tail = ++tail;
    }
    void flush() {
        if (n != 0) {
            emit();
            n = 0;
        }
    }
    void close() {
        flush();
        ctrl()->dropped_sync = dropped;
        std::atomic_thread_fence(std::memory_order_release);
        ctrl()->done = kernel_profiler::kRelayDoneWord;
    }
    __attribute__((noinline)) void add(
        uint64_t r, uint64_t w8, uint32_t k8, uint32_t slope, kernel_profiler::SyncMeta record_meta) {
        if (n != 0) {
            if (kernel_profiler::word_of(record_meta) == kernel_profiler::word_of(meta)) {
                const int32_t off = static_cast<int32_t>(
                    static_cast<uint32_t>(w8) - static_cast<uint32_t>(w0) -
                    rates.slope * static_cast<uint32_t>(r - r0));
                if (kernel_profiler::sync_local_step_fits(r - r0, off)) {
                    steps[n - 1] = kernel_profiler::word_of(
                        kernel_profiler::SyncLocalStep{.refclk = static_cast<uint32_t>(r - r0), .wall_off = off});
                    rates.k8[n] = static_cast<uint8_t>(k8);
                    if (++n == kernel_profiler::kSyncLocalPoints) {
                        flush();
                    }
                    return;
                }
            }
            flush();
        }
        r0 = r;
        w0 = w8;
        rates.k8[0] = static_cast<uint8_t>(k8);
        rates.slope = static_cast<uint8_t>(slope);
        meta = record_meta;
        n = 1;
    }
};

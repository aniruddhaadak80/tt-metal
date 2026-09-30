// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// What the multicast and ping-pong tests share: opening the system mesh, the matmul and NoC load kernels with their
// calibration against the device, and the per-core flag a kernel sets when it stops waiting for a peer.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "hostdev/streaming_profiler_common.h"

namespace streaming_profiler_workload {

inline constexpr std::string_view kKernelDir = "tests/tt_metal/tools/profiler/streaming_profiler_sync/kernels/";

constexpr uint32_t kLoadBytes = 8192;
constexpr uint32_t kLoadPartners = 8;
constexpr uint32_t kLoadTileBytes = 2048;

// One region of every worker's L1, from the allocator: the kernels' flag, where word 0 is the flag and word 1 the round
// a kernel gave up waiting on (0 if none), the multicast's round values and the load's scratch.
struct WorkerL1 {
    static constexpr uint32_t kFlagBytes = 64;
    static constexpr uint32_t kValuesBytes = 64 * 1024;
    static constexpr uint32_t kLoadScratchBytes = 4 * kLoadBytes;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer;
    uint32_t flag() const { return static_cast<uint32_t>(buffer->address()); }
    uint32_t values() const { return flag() + kFlagBytes; }
    uint32_t load_scratch() const { return values() + kValuesBytes; }
};

// One page per L1 bank, and every worker is one bank, so each worker holds the region at the same address.
inline WorkerL1 reserve_worker_l1(tt::tt_metal::distributed::MeshDevice& mesh) {
    constexpr uint32_t kPerCore = WorkerL1::kFlagBytes + WorkerL1::kValuesBytes + WorkerL1::kLoadScratchBytes;
    const uint32_t banks = mesh.allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
    return {tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = uint64_t{kPerCore} * banks},
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = kPerCore, .buffer_type = tt::tt_metal::BufferType::L1},
        &mesh)};
}

using Clock = std::chrono::steady_clock;
inline double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

inline double run_once(tt::tt_metal::distributed::MeshCommandQueue& cq, tt::tt_metal::distributed::MeshWorkload& w) {
    const auto t0 = Clock::now();
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, w, /*blocking=*/false);
    tt::tt_metal::distributed::Finish(cq);
    return seconds_since(t0);
}

inline std::shared_ptr<tt::tt_metal::distributed::MeshDevice> open_system_mesh(const char* tag) {
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create(
        tt::tt_metal::distributed::MeshDeviceConfig(tt::tt_metal::distributed::SystemMesh::instance().shape()),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_command_queues=*/1);
    if (!tt::tt_metal::experimental::streaming_profiler::IsActive()) {
        std::fprintf(stderr, "[%s] needs TT_METAL_STREAMING_PROFILER=1\n", tag);
        mesh_device->close();
        return nullptr;
    }
    return mesh_device;
}

inline tt::tt_metal::CoreRange worker_grid(tt::tt_metal::distributed::MeshDevice& mesh) {
    const tt::tt_metal::CoreCoord grid = mesh.compute_with_storage_grid_size();
    return tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(grid.x - 1, grid.y - 1));
}

struct LoadSpec {
    uint32_t bursts = 1, mm_iters = 0, dm_iters = 0, idle_cycles = 0;
};

inline void add_load(
    tt::tt_metal::Program& program,
    tt::tt_metal::distributed::MeshDevice& mesh,
    const WorkerL1& l1,
    const tt::tt_metal::CoreRangeSet& cores,
    const LoadSpec& s,
    const std::set<tt::tt_metal::CoreCoord>& avoid = {}) {
    const std::string kernels(kKernelDir);
    const tt::tt_metal::CoreCoord grid = mesh.compute_with_storage_grid_size();
    for (const tt::CBIndex cb : {tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16}) {
        tt::tt_metal::CreateCircularBuffer(
            program,
            cores,
            tt::tt_metal::CircularBufferConfig(2 * kLoadTileBytes, {{cb, tt::DataFormat::Float16_b}})
                .set_page_size(cb, kLoadTileBytes));
    }
    const tt::tt_metal::KernelHandle compute = tt::tt_metal::CreateKernel(
        program,
        kernels + "load_compute.cpp",
        cores,
        tt::tt_metal::ComputeConfig{.math_fidelity = tt::tt_metal::MathFidelity::HiFi4});
    const tt::tt_metal::KernelHandle dm0 = tt::tt_metal::CreateKernel(
        program,
        kernels + "load_dm.cpp",
        cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
    const tt::tt_metal::KernelHandle dm1 = tt::tt_metal::CreateKernel(
        program,
        kernels + "load_dm.cpp",
        cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::RISCV_1_default});
    for (const tt::tt_metal::CoreRange& r : cores.ranges()) {
        for (const tt::tt_metal::CoreCoord& c : r) {
            tt::tt_metal::SetRuntimeArgs(program, compute, c, {s.bursts, s.mm_iters, s.idle_cycles});
            std::vector<uint32_t> args = {
                l1.load_scratch(), kLoadBytes, s.bursts, s.dm_iters, s.idle_cycles, kLoadPartners};
            for (uint32_t p = 1; p <= kLoadPartners; p++) {
                tt::tt_metal::CoreCoord l((c.x + 3 * p) % grid.x, (c.y + 5 * p) % grid.y);
                while (avoid.contains(l)) {
                    l.x = (l.x + 1) % grid.x;
                }
                const tt::tt_metal::CoreCoord v = mesh.worker_core_from_logical_core(l);
                args.push_back(kernel_profiler::word_of(
                    kernel_profiler::NocXy{.x = static_cast<uint32_t>(v.x), .y = static_cast<uint32_t>(v.y)}));
            }
            tt::tt_metal::SetRuntimeArgs(program, dm0, c, args);
            tt::tt_metal::SetRuntimeArgs(program, dm1, c, args);
        }
    }
}

inline tt::tt_metal::distributed::MeshWorkload make_load(
    tt::tt_metal::distributed::MeshDevice& mesh,
    const WorkerL1& l1,
    const tt::tt_metal::CoreRangeSet& cores,
    const LoadSpec& s) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    add_load(program, mesh, l1, cores, s);
    tt::tt_metal::distributed::MeshWorkload w;
    w.add_program(tt::tt_metal::distributed::MeshCoordinateRange(mesh.shape()), std::move(program));
    return w;
}

// Seconds per HiFi4 matmul block and per NoC load round on the whole worker grid, each timed on its second run.
struct LoadRate {
    double mm_s = 0.0, dm_s = 0.0;
    // `base` with iteration counts set so each RISC's load runs for about `seconds`.
    LoadSpec lasting(double seconds, LoadSpec base = {}) const {
        base.mm_iters = std::max<uint32_t>(1, static_cast<uint32_t>(std::lround(seconds / mm_s)));
        base.dm_iters = std::max<uint32_t>(1, static_cast<uint32_t>(std::lround(seconds / dm_s)));
        return base;
    }
};

inline LoadRate calibrate_load(tt::tt_metal::distributed::MeshDevice& mesh, const WorkerL1& l1) {
    const auto timed = [&](const LoadSpec& spec) {
        tt::tt_metal::distributed::MeshWorkload w = make_load(mesh, l1, worker_grid(mesh), spec);
        run_once(mesh.mesh_command_queue(), w);
        return run_once(mesh.mesh_command_queue(), w);
    };
    const LoadSpec mm{.mm_iters = 2000}, dm{.dm_iters = 200};
    return {.mm_s = timed(mm) / mm.mm_iters, .dm_s = timed(dm) / dm.dm_iters};
}

struct FlagCore {
    tt::tt_metal::IDevice* device;
    tt::tt_metal::CoreCoord core;
};

inline void clear_flags(const WorkerL1& l1, const std::vector<FlagCore>& cores) {
    std::vector<uint32_t> zero = {0, 0};
    for (const FlagCore& c : cores) {
        tt::tt_metal::detail::WriteToDeviceL1(c.device, c.core, l1.flag(), zero);
    }
}

inline uint32_t count_gave_up(const WorkerL1& l1, const std::vector<FlagCore>& cores, const char* tag) {
    uint32_t gave_up = 0;
    for (const FlagCore& c : cores) {
        std::vector<uint32_t> words(2, 0);
        tt::tt_metal::detail::ReadFromDeviceL1(c.device, c.core, l1.flag(), 8, words);
        if (words[1] != 0) {
            std::printf(
                "[%s] chip %d core (%zu,%zu) gave up waiting for round %u (flag %u)\n",
                tag,
                c.device->id(),
                c.core.x,
                c.core.y,
                words[1],
                words[0]);
            gave_up++;
        }
    }
    return gave_up;
}

}  // namespace streaming_profiler_workload

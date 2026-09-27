// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/matmul/sfpu_activation_helpers.hpp"

namespace compute_kernel_lib {
namespace matmul_config {

// WaitAndRetainOnLastBlock leaves the last K block for caller reuse.
enum class InputPolicy : uint8_t { WaitAndPopPerKBlock, WaitAndRetainOnLastBlock };

// Controls entry initialization; both modes restore matmul state after partial reloads.
enum class InitMode : uint8_t { Initialize, AssumeInitialized };

// Configure input and accumulation pack formats, or retain caller-provided state.
enum class DataFormatReconfig : uint8_t { InputAndOutput, None };

}  // namespace matmul_config

/**
 * Block dimensions in tiles: M = in0_num_subblocks * out_subblock_h,
 * N = in1_num_subblocks * out_subblock_w, K = num_k_blocks * in0_block_k.
 */
struct MatmulShape {
    uint32_t in0_num_subblocks;  // Output subblock count along M.
    uint32_t in1_num_subblocks;  // Output subblock count along N.
    uint32_t out_subblock_h;     // Output subblock height in tiles.
    uint32_t out_subblock_w;     // Output subblock width in tiles.
    uint32_t in0_block_k;        // Tiles per K block.
    uint32_t num_k_blocks;       // K block count.
    uint32_t batch = 1;          // Independent batch slices.

    // Valid columns in the last in1 subblock; 0 uses out_subblock_w. Packing stays full-width.
    uint32_t last_in1_subblock_w_valid = 0;

    // Producer N stride in tiles; 0 uses in1_num_subblocks * out_subblock_w.
    uint32_t in1_per_core_w = 0;

    // UnpackToDestFp32 reload view; UINT32_MAX uses interm.
    uint32_t partials_reload_cb_id = UINT32_MAX;

    static constexpr MatmulShape of(
        uint32_t in0_num_subblocks,
        uint32_t in1_num_subblocks,
        uint32_t out_subblock_h,
        uint32_t out_subblock_w,
        uint32_t in0_block_k,
        uint32_t num_k_blocks,
        uint32_t batch = 1,
        uint32_t in1_per_core_w = 0) {
        return {
            in0_num_subblocks,
            in1_num_subblocks,
            out_subblock_h,
            out_subblock_w,
            in0_block_k,
            num_k_blocks,
            batch,
            /*last_in1_subblock_w_valid=*/0,
            in1_per_core_w};
    }
};

// Compile-time counterpart of MatmulShape.
template <
    uint32_t In0NumSubblocks,
    uint32_t In1NumSubblocks,
    uint32_t OutSubblockH,
    uint32_t OutSubblockW,
    uint32_t In0BlockK,
    uint32_t NumKBlocks,
    uint32_t Batch = 1,
    uint32_t LastIn1SubblockWValid = 0,
    uint32_t In1PerCoreW = 0>
struct StaticMatmulShape {
    static constexpr uint32_t in0_num_subblocks = In0NumSubblocks;
    static constexpr uint32_t in1_num_subblocks = In1NumSubblocks;
    static constexpr uint32_t out_subblock_h = OutSubblockH;
    static constexpr uint32_t out_subblock_w = OutSubblockW;
    static constexpr uint32_t in0_block_k = In0BlockK;
    static constexpr uint32_t num_k_blocks = NumKBlocks;
    static constexpr uint32_t batch = Batch;
    static constexpr uint32_t last_in1_subblock_w_valid = LastIn1SubblockWValid;
    static constexpr uint32_t in1_per_core_w = In1PerCoreW;
    uint32_t partials_reload_cb_id = UINT32_MAX;
};

struct NoPostCompute {
    ALWI void operator()(uint32_t) const {}
};

struct NoPreKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {}
};

/**
 * Subblocked matmul with software or packer L1 accumulation.
 * Shape accepts MatmulShape or StaticMatmulShape.
 *
 * Call compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out) once at startup.
 * AssumeInitialized also requires matmul_block_init with matching operands and shape.
 * Initialize SFPU activations at startup with ActivationInitHelper::init().
 *
 * Pass CB IDs. in0, in1, and out must be distinct; each K block consumes
 * M x block_k input tiles from in0 and block_k x N from in1, subject to InputPolicy.
 * out and interm may share compatible L1 storage. The same CB ID is allowed only
 * for a kernel-local result with no concurrent consumer. L1 accumulation requires
 * exactly one output block of partials capacity. For one K block, interm may be out.
 *
 * Output is contiguous within each subblock, in subblock traversal order.
 * Use reblock_and_untilize for untilized output. TransposeIn1 transposes B tiles,
 * not the tile grid, which the caller must arrange.
 *
 * Activation runs on completed sums on the packer thread; ReluActivation uses
 * hardware packer ReLU. Defer activation when bias or another operation must precede it.
 *
 * PreKBlockFn(block, num_k_blocks, is_last) runs before input waits and must
 * restore matmul state after preprocessing. PostComputeFn(num_tiles) runs on math
 * on the completed sum before packing the last K block.
 *
 * Avoid HiFi4 with BF16 inputs and FP32 DST on Wormhole B0 (issue #38306).
 * SKIP_COMPUTE skips the matmul LLK call but retains synchronization.
 */
template <
    bool TransposeIn1 = false,
    bool PackerL1Acc = false,
    matmul_config::InitMode InitMode = matmul_config::InitMode::Initialize,
    matmul_config::InputPolicy InputPolicy = matmul_config::InputPolicy::WaitAndPopPerKBlock,
    matmul_config::DataFormatReconfig Reconfig = matmul_config::DataFormatReconfig::InputAndOutput,
    typename Activation = NoneActivation,
    typename PostComputeFn = NoPostCompute,
    typename PreKBlockFn = NoPreKBlock,
    typename Shape>
ALWI void matmul(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t out_cb_id,
    uint32_t interm_cb_id,
    const Shape& shape,
    PostComputeFn post_compute = {},
    PreKBlockFn pre_k_block = {});

}  // namespace compute_kernel_lib

#include "matmul.inl"

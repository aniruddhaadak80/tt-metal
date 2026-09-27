// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Blocked matmul with optional transpose, bias, activation, and untilize.

#include <cstdint>
#include "experimental/kernel_args.h"
#include <type_traits>

#include "ttnn/cpp/ttnn/kernel_lib/matmul/matmul.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul/reblock_untilize_helpers.hpp"

#ifdef FUSE_BIAS
#include "ttnn/cpp/ttnn/kernel_lib/matmul/bias_add_helpers.hpp"
#endif

#include "api/compute/eltwise_binary.h"
#include "api/compute/transpose.h"
#include "api/compute/compute_kernel_hw_startup.h"

#ifdef SFPU_ACTIVATION
#include "ttnn/cpp/ttnn/kernel_lib/matmul/sfpu_activation_helpers.hpp"
#endif

namespace {

// Transpose input tiles in DST-sized chunks, including a final partial chunk.
template <uint32_t in_block_num_tiles, uint32_t block_size = 4>
FORCE_INLINE void transpose_tile_block(DataflowBuffer& in_transpose_buf, DataflowBuffer& in_buf) {
    constexpr uint32_t num_blocks = in_block_num_tiles / block_size;
    constexpr uint32_t last_block_size = in_block_num_tiles % block_size;

    const uint32_t in_transpose_cb_id = in_transpose_buf.get_id();
    const uint32_t in_cb_id = in_buf.get_id();

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        in_transpose_buf.wait_front(block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            transpose_tile(in_transpose_cb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in_transpose_buf.pop_front(block_size);

        in_buf.reserve_back(block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < block_size; tile_idx++) {
            pack_tile(tile_idx, in_cb_id);
        }
        tile_regs_release();
        in_buf.push_back(block_size);
    }

    if constexpr (last_block_size > 0) {
        in_transpose_buf.wait_front(last_block_size);
        tile_regs_acquire();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            transpose_tile(in_transpose_cb_id, tile_idx, tile_idx);
        }
        tile_regs_commit();
        in_transpose_buf.pop_front(last_block_size);

        in_buf.reserve_back(last_block_size);
        tile_regs_wait();
        for (uint32_t tile_idx = 0; tile_idx < last_block_size; tile_idx++) {
            pack_tile(tile_idx, in_cb_id);
        }
        tile_regs_release();
        in_buf.push_back(last_block_size);
    }
}

}  // namespace

// Please update
// tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/bmm_large_block_zm_fused_bias_activation_copy.cpp
// when making any changes to this file.
// Have to keep a copy because cannot import ttnn into tests/tt_metal.
// With FUSE_BIAS: row_broadcast_bias (row-broadcast vs elementwise add_tiles) is compile-time arg 18 here;
// the perf copy uses index 14 (different compile-time arg layout).

void kernel_main() {
    using namespace compute_kernel_lib;

#ifdef MATMUL_DRAM_SHARDED
    if (get_arg(args::is_worker_core) != 1) {
        return;
    }
#endif

    // Compile-time arguments
    constexpr uint32_t in0_block_w = get_arg(args::in0_block_w);
    constexpr uint32_t in0_num_subblocks = get_arg(args::in0_num_subblocks);
    constexpr uint32_t in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    [[maybe_unused]] constexpr uint32_t in0_subblock_num_tiles = get_arg(args::in0_subblock_num_tiles);
    constexpr uint32_t in1_num_subblocks = get_arg(args::in1_num_subblocks);
    [[maybe_unused]] constexpr uint32_t in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    constexpr uint32_t in1_block_w = get_arg(args::in1_block_w);
    constexpr uint32_t num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr uint32_t num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr uint32_t num_blocks_h_dim = get_arg(args::num_blocks_h_dim);
    constexpr uint32_t out_subblock_h = get_arg(args::out_subblock_h);
    constexpr uint32_t out_subblock_w = get_arg(args::out_subblock_w);
    constexpr uint32_t out_subblock_num_tiles = get_arg(args::out_subblock_num_tiles);
    constexpr uint32_t batch = get_arg(args::batch);
    [[maybe_unused]] constexpr uint32_t out_block_num_tiles = get_arg(args::out_block_num_tiles);
    constexpr bool untilize_out = get_arg(args::untilize_out);
    constexpr bool get_batch_from_reader = (bool)get_arg(args::get_batch_from_reader);
#ifdef IN0_TRANSPOSE_TILE
    constexpr bool in0_transpose_tile = true;
#else
    constexpr bool in0_transpose_tile = false;
#endif

    constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;

    // Circular buffer IDs
#ifdef IN0_TRANSPOSE_TILE
    constexpr uint32_t in0_cb_id = dfb::in0_transposed;
#else
    constexpr uint32_t in0_cb_id = dfb::in0;
#endif
    constexpr uint32_t in1_cb_id = dfb::in1;
    constexpr uint32_t out_cb_id = dfb::out;
    constexpr uint32_t mm_partials_cb_id = dfb::intermed0;
    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? mm_partials_cb_id : out_cb_id;
    constexpr uint32_t in0_transpose_cb_id = dfb::in0;

    DataflowBuffer in0_buf(in0_cb_id);

    DataflowBuffer out_buf(out_cb_id);
    DataflowBuffer mm_partials_buf(mm_partials_cb_id);
    DataflowBuffer untilize_mode_out_buf(untilize_mode_out_cb_id);
    DataflowBuffer in0_transpose_buf(in0_transpose_cb_id);

#ifdef FUSE_BIAS
    constexpr uint32_t bias_cb_id = dfb::bias;
    constexpr uint32_t bias_ntiles = get_arg(args::bias_ntiles);
    // true: row-0 broadcast ([N] / [...,1,N]); false: elementwise add_tiles (bias has multiple M rows).
    constexpr bool row_broadcast_bias = (bool)get_arg(args::row_broadcast_bias);
    DataflowBuffer bias_buf(bias_cb_id);
#endif

    // Default activation parameters keep helper instantiations valid when activation is disabled.
#ifdef SFPU_ACTIVATION
    constexpr KernelActivation activation_type = static_cast<KernelActivation>(get_arg(args::activation_type));
    constexpr uint32_t activation_param0 = get_arg(args::activation_param0);
    constexpr uint32_t activation_param1 = get_arg(args::activation_param1);
    constexpr uint32_t activation_param2 = get_arg(args::activation_param2);
#else
    constexpr KernelActivation activation_type = KernelActivation::NONE;
    constexpr uint32_t activation_param0 = 0;
    constexpr uint32_t activation_param1 = 0;
    constexpr uint32_t activation_param2 = 0;
#endif

    // Feature flags
#ifdef IN1_TRANSPOSE_TILE
    constexpr uint32_t in1_transpose_tile = true;
#else
    constexpr uint32_t in1_transpose_tile = false;
#endif

    constexpr bool l1_acc =
#ifdef PACKER_L1_ACC
        true;
#else
        false;
#endif

#if defined(PACK_RELU) && !defined(FUSE_BIAS)
    constexpr bool matmul_pack_relu = !untilize_out;
#else
    constexpr bool matmul_pack_relu = false;
#endif

    // Apply activation after bias when bias is fused.
#ifdef FUSE_BIAS
    constexpr KernelActivation matmul_activation = KernelActivation::NONE;
    constexpr KernelActivation bias_activation = activation_type;
#else
    constexpr KernelActivation matmul_activation = untilize_out ? KernelActivation::NONE : activation_type;
    constexpr KernelActivation bias_activation = KernelActivation::NONE;
#endif

    // Retain matmul state across calls for heterogeneous-tile DRAM-sharded inputs.
    // Initialize SFPU activation once at startup.
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb_id, in1_cb_id, mm_partials_cb_id);
    matmul_block_init(in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
    if constexpr (activation_type != KernelActivation::NONE) {
        ActivationInitHelper<activation_type, activation_param0, activation_param1>::init();
    }

    // Main loop: batch × output blocks
    for (uint32_t b = 0; b < batch; b++) {
        if constexpr (get_batch_from_reader) {
            bool is_batch_valid = false;
            UNPACK(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
            MATH(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
            PACK(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
            if (!is_batch_valid) {
                continue;
            }
        }

        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                // Reset packer state for this output block
#ifdef PACK_RELU
                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((llk_pack_relu_config(ReluConfig::none())));
                }
#endif
                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((pack_reconfig_data_format(mm_partials_cb_id)));
                }

                // Phase 1: K-loop matmul
#ifdef FUSE_BIAS
                constexpr uint32_t phase1_out_cb_id = mm_partials_cb_id;
                // Partials may share storage with output still being drained by the writer.
                if constexpr (!l1_acc && num_blocks_inner_dim > 1) {
                    out_buf.reserve_back(out_block_num_tiles);
                }
#else
                constexpr uint32_t phase1_out_cb_id = untilize_mode_out_cb_id;
#endif
                auto shape = MatmulShape::of(
                    in0_num_subblocks,
                    in1_num_subblocks,
                    out_subblock_h,
                    out_subblock_w,
                    in0_block_w,
                    num_blocks_inner_dim,
                    /*batch=*/1,
                    /*in1_per_core_w=*/in1_block_w);
#ifdef MM_PARTIALS_RELOAD_ALIAS
                shape.partials_reload_cb_id = dfb::intermed0_reload_alias;
#endif
#ifdef MATMUL_DRAM_SHARDED
                // The final DRAM-sharded subblock may have fewer input columns than its padded output width.
                shape.last_in1_subblock_w_valid = get_arg(args::last_subblock_w_valid);
#endif
                if constexpr (in0_transpose_tile) {
                    auto xpose = [&](uint32_t /*k_iter*/, uint32_t /*num_k_iters*/, bool /*is_last*/) {
                        reconfig_data_format_srca(in1_cb_id, in0_transpose_cb_id);
                        transpose_init(in0_transpose_cb_id);
                        PACK((pack_reconfig_data_format(in0_cb_id)));
#ifdef PACKER_L1_ACC
                        PACK((llk_pack_reconfig_l1_acc(0)));
#endif
                        transpose_tile_block<in0_block_num_tiles>(in0_transpose_buf, in0_buf);
                        reconfig_data_format_srca(in0_transpose_cb_id, in1_cb_id);
                        matmul_block_init(
                            in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                        PACK((pack_reconfig_data_format(mm_partials_cb_id)));
                    };
                    compute_kernel_lib::matmul<
                        in1_transpose_tile,
                        l1_acc,
                        matmul_config::InitMode::AssumeInitialized,
                        matmul_config::InputPolicy::WaitAndPopPerKBlock,
                        matmul_config::DataFormatReconfig::None,
                        ActivationOp<
                            matmul_activation,
                            activation_param0,
                            activation_param1,
                            activation_param2,
                            matmul_pack_relu>>(
                        in0_cb_id, in1_cb_id, phase1_out_cb_id, mm_partials_cb_id, shape, NoPostCompute{}, xpose);
                } else {
                    compute_kernel_lib::matmul<
                        in1_transpose_tile,
                        l1_acc,
                        matmul_config::InitMode::AssumeInitialized,
                        matmul_config::InputPolicy::WaitAndPopPerKBlock,
                        matmul_config::DataFormatReconfig::None,
                        ActivationOp<
                            matmul_activation,
                            activation_param0,
                            activation_param1,
                            activation_param2,
                            matmul_pack_relu>>(in0_cb_id, in1_cb_id, phase1_out_cb_id, mm_partials_cb_id, shape);
                }

                // Phase 2: Bias addition
#ifdef FUSE_BIAS
#ifdef PACK_RELU
                // if last block we pack the final result with relu enabled
                PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                PACK((pack_reconfig_data_format(out_cb_id)));
#endif
#ifdef PACKER_L1_ACC
                PACK((llk_pack_reconfig_l1_acc(0)));
#endif

                // Retain bias across batches and height blocks when the reader publishes it only once.
                if ((b == 0 && bh == 0) || num_blocks_w_dim > 1) {
                    bias_buf.wait_front(bias_ntiles);
                }

#ifdef BIAS_FULL_BLOCK
                // Full-block elementwise bias needs (M, N) indexing; the shared helper indexes bias by N only.
                static_assert(
                    !row_broadcast_bias, "BIAS_FULL_BLOCK implies elementwise bias (row_broadcast_bias must be false)");
#ifdef MATMUL_DRAM_SHARDED
                constexpr uint32_t bias_last_subblock_w_valid = get_arg(args::last_subblock_w_valid);
#else
                constexpr uint32_t bias_last_subblock_w_valid = out_subblock_w;
#endif
                constexpr bool bias_last_subblock_padded = bias_last_subblock_w_valid < out_subblock_w;

                reconfig_data_format(in1_cb_id, mm_partials_cb_id, in0_cb_id, bias_cb_id);
                add_init(mm_partials_cb_id, bias_cb_id);
                for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                    int in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                        // Last in1 subblock may have padded lanes whose bias tile was never pushed;
                        // redirect those out-of-range reads to tile 0 (dropped by the writer).
                        const bool is_last_in1_subblock_padded =
                            bias_last_subblock_padded && (in1_subblock == in1_num_subblocks - 1);
                        mm_partials_buf.wait_front(out_subblock_num_tiles);
                        tile_regs_acquire();
                        for (uint32_t i = 0, j = 0; j < out_subblock_h; j++) {
                            // The bias CB holds a full [M, N] tile block; bias_tile_idx is (row m_tile,
                            // column in1_index_subblock_offset) within it.
                            const uint32_t m_tile = in0_subblock * out_subblock_h + j;
                            uint32_t bias_tile_idx = m_tile * in1_block_w + in1_index_subblock_offset;
                            for (uint32_t k = 0; k < out_subblock_w; k++, i++) {
                                const uint32_t safe_bias_tile_idx =
                                    (is_last_in1_subblock_padded && k >= bias_last_subblock_w_valid) ? 0u
                                                                                                     : bias_tile_idx;
                                add_tiles(mm_partials_cb_id, bias_cb_id, i, safe_bias_tile_idx, i);
                                bias_tile_idx++;
                            }
                        }
                        tile_regs_commit();
                        mm_partials_buf.pop_front(out_subblock_num_tiles);

                        // Pack out to output buffer
                        untilize_mode_out_buf.reserve_back(out_subblock_num_tiles);
#ifdef SFPU_ACTIVATION
                        apply_activation_from_pack<
                            activation_type,
                            activation_param0,
                            activation_param1,
                            activation_param2>(out_subblock_num_tiles);
#else
                        tile_regs_wait();
#endif
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, untilize_mode_out_cb_id);
                        }
                        tile_regs_release();
                        untilize_mode_out_buf.push_back(out_subblock_num_tiles);

                        in1_index_subblock_offset += out_subblock_w;
                    }
                }
#else
                constexpr BiasBroadcast bias_broadcast =
                    row_broadcast_bias ? BiasBroadcast::RowBroadcast : BiasBroadcast::Elementwise;
                const auto bias_shape =
                    BiasAddShape::of(in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w);
                add_bias_bcast_rows<
                    bias_broadcast,
                    bias_add_config::NoPostBias,
                    ActivationOp<bias_activation, activation_param0, activation_param1, activation_param2>>(
                    mm_partials_cb_id, bias_cb_id, untilize_mode_out_cb_id, bias_shape);
#endif  // BIAS_FULL_BLOCK

                if constexpr (num_blocks_w_dim > 1) {
                    bias_buf.pop_front(bias_ntiles);
                }
#endif  // FUSE_BIAS

                // Phase 3: Untilize
                if constexpr (untilize_out) {
#ifdef PACK_RELU
                    PACK((llk_pack_relu_config(ReluConfig::none())));
#endif  // PACK_RELU
#ifndef FUSE_BIAS
                    reconfig_data_format_srca(in1_cb_id, mm_partials_cb_id);
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                    PACK((pack_reconfig_data_format(out_cb_id)));
#endif
#ifdef PACKER_L1_ACC
                    PACK((llk_pack_reconfig_l1_acc(0)));
#endif
#endif  // !FUSE_BIAS
        // Reblock the subblock stream before untilizing.

                    reblock_and_untilize<
                        out_subblock_w,
                        out_block_w,
                        /*reconfigure=*/false>(in0_num_subblocks, out_subblock_h, mm_partials_cb_id, out_cb_id);
                }

                // Reconfigure for next output block
                if constexpr (batch > 1 || num_blocks_w_dim > 1 || num_blocks_h_dim > 1) {
#ifdef FUSE_BIAS
                    reconfig_data_format(mm_partials_cb_id, in1_cb_id, bias_cb_id, in0_cb_id);
#else
                    reconfig_data_format_srca(mm_partials_cb_id, in1_cb_id);
#endif
                    matmul_block_init(
                        in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
                }
            }
        }
    }
#ifdef FUSE_BIAS
    // Release resident bias after its final use; multi-width-block bias is popped per block above.
    if constexpr (num_blocks_w_dim == 1) {
        bias_buf.pop_front(bias_ntiles);
    }
#endif
}

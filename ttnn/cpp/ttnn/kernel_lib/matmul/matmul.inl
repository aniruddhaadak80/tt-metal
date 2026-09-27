// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dfb_helpers_compute.hpp"

namespace compute_kernel_lib {
template <
    bool TransposeIn1,
    bool PackerL1Acc,
    matmul_config::InitMode InitMode,
    matmul_config::InputPolicy InputPolicy,
    matmul_config::DataFormatReconfig Reconfig,
    typename Activation,
    typename PostComputeFn,
    typename PreKBlockFn,
    typename Shape>
ALWI void matmul(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t out_cb_id,
    uint32_t interm_cb_id,
    const Shape& shape,
    PostComputeFn post_compute,
    PreKBlockFn pre_k_block) {
    constexpr bool needs_postprocess = Activation::pack_relu || Activation::activation != KernelActivation::NONE;

    DataflowBuffer in0_buf(in0_cb_id), in1_buf(in1_cb_id), out_buf(out_cb_id), interm_buf(interm_cb_id);
    const bool output_is_interm = out_cb_id == interm_cb_id;
    const uint32_t reload_cb_id =
        shape.partials_reload_cb_id == UINT32_MAX ? interm_cb_id : shape.partials_reload_cb_id;

    const bool reload_last = !output_is_interm || needs_postprocess || !std::is_same_v<PostComputeFn, NoPostCompute>;

    ASSERT(shape.in0_block_k > 0);
    ASSERT(shape.in0_num_subblocks > 0);
    ASSERT(shape.in1_num_subblocks > 0);
    ASSERT(shape.num_k_blocks > 0);
    ASSERT(shape.out_subblock_h > 0);
    ASSERT(shape.out_subblock_w > 0);
    ASSERT(shape.batch > 0);
    ASSERT(in0_cb_id != out_cb_id);
    ASSERT(in1_cb_id != out_cb_id);
    ASSERT(shape.out_subblock_h * shape.out_subblock_w <= compute_kernel_lib::DEST_AUTO_LIMIT);
    if constexpr (Reconfig == matmul_config::DataFormatReconfig::InputAndOutput) {
        // Matmul convention: srca takes in1, srcb takes in0.
        reconfig_data_format(in1_cb_id, in0_cb_id);
        PACK((pack_reconfig_data_format(interm_cb_id)));
    }
    if constexpr (InitMode == matmul_config::InitMode::Initialize) {
        matmul_block_init(
            in0_cb_id, in1_cb_id, TransposeIn1, shape.out_subblock_w, shape.out_subblock_h, shape.in0_block_k);
    }

    const uint32_t out_subblock_num_tiles = shape.out_subblock_h * shape.out_subblock_w;
    const uint32_t in0_subblock_num_tiles = shape.out_subblock_h * shape.in0_block_k;
    const uint32_t in0_block_num_tiles = in0_subblock_num_tiles * shape.in0_num_subblocks;
    uint32_t in1_per_core_w = shape.in1_per_core_w;
    if (in1_per_core_w == 0) {
        in1_per_core_w = shape.out_subblock_w * shape.in1_num_subblocks;
    }
    const uint32_t in1_block_num_tiles = in1_per_core_w * shape.in0_block_k;
    const uint32_t out_block_num_tiles = out_subblock_num_tiles * shape.in0_num_subblocks * shape.in1_num_subblocks;

    for (uint32_t b = 0; b < shape.batch; b++) {
        for (uint32_t block = 0; block < shape.num_k_blocks; block++) {
            const bool last_out = block == (shape.num_k_blocks - 1);
            const bool enable_reload = block != 0 && (!PackerL1Acc || (last_out && reload_last));

            pre_k_block(block, shape.num_k_blocks, last_out);

            in0_buf.wait_front(in0_block_num_tiles);
            in1_buf.wait_front(in1_block_num_tiles);

            DataflowBuffer& dst_buf = last_out ? out_buf : interm_buf;
            const uint32_t dst_cb_id = dst_buf.get_id();

            if constexpr (PackerL1Acc) {
                PACK((pack_reconfig_data_format(dst_cb_id)));
                PACK((llk_pack_reconfig_l1_acc(block != 0 && !enable_reload)));
            }

            if constexpr (Activation::pack_relu) {
                if (last_out) {
                    PACK((llk_pack_relu_config(ReluConfig::zero())));
                }
            }

            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < shape.in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < shape.in1_num_subblocks; in1_subblock++) {
                    tile_regs_acquire();

                    // Narrow the final FMA width without changing the packed subblock size.
                    const uint32_t effective_subblock_w =
                        (shape.last_in1_subblock_w_valid != 0 && in1_subblock == shape.in1_num_subblocks - 1)
                            ? shape.last_in1_subblock_w_valid
                            : shape.out_subblock_w;

                    if (enable_reload) {
                        reconfig_data_format_srca(in1_cb_id, reload_cb_id);
                        copy_init(reload_cb_id);
                        if (reload_cb_id != interm_cb_id) {
                            DataflowBuffer source(interm_cb_id), reload(reload_cb_id);
                            UNPACK((reload.evil_set_read_ptr(source.get_read_ptr())));
                        }

                        interm_buf.wait_front(out_subblock_num_tiles);
                        copy_block(reload_cb_id, 0, 0, out_subblock_num_tiles);
                        interm_buf.pop_front(out_subblock_num_tiles);

#ifndef ARCH_QUASAR
                        reconfig_data_format_srca(reload_cb_id, in1_cb_id);
                        matmul_block_init(
                            in0_cb_id,
                            in1_cb_id,
                            TransposeIn1,
                            shape.out_subblock_w,
                            shape.out_subblock_h,
                            shape.in0_block_k);
#endif
                    }

                    uint32_t in0_index = in0_index_subblock_offset;
                    uint32_t in1_index = in1_index_subblock_offset;
                    for (uint32_t inner_dim = 0; inner_dim < shape.in0_block_k; inner_dim++) {
#ifndef SKIP_COMPUTE
                        ckernel::matmul_block(
                            in0_cb_id,
                            in1_cb_id,
                            in0_index,
                            in1_index,
                            0,
                            TransposeIn1,
                            effective_subblock_w,
                            shape.out_subblock_h,
                            shape.in0_block_k);
#else
                        (void)in0_index;
                        (void)in1_index;
#endif
                        in0_index++;
                        in1_index += in1_per_core_w;
                    }

                    if (last_out) {
                        post_compute(out_subblock_num_tiles);
                    }
                    tile_regs_commit();
                    if constexpr (!PackerL1Acc && get_fp32_dest_acc_enabled()) {
                        PACK((pack_reconfig_data_format(dst_cb_id)));
                    }
                    // Partials may alias output still being consumed by the writer.
                    if (!output_is_interm && block == 0 && !last_out) {
                        const uint32_t tiles_to_wait =
                            (in0_subblock * shape.in1_num_subblocks + in1_subblock + 1) * out_subblock_num_tiles;
                        out_buf.reserve_back(tiles_to_wait);
                    }
                    dst_buf.reserve_back(out_subblock_num_tiles);
                    if constexpr (Activation::activation != KernelActivation::NONE) {
                        if (last_out) {
                            apply_activation_from_pack<
                                Activation::activation,
                                Activation::param0,
                                Activation::param1,
                                Activation::param2>(out_subblock_num_tiles);
                        } else {
                            tile_regs_wait();
                        }
                    } else {
                        tile_regs_wait();
                    }
                    pack_block(0, dst_cb_id, out_subblock_num_tiles);

                    tile_regs_release();
                    dst_buf.push_back(out_subblock_num_tiles);

                    in1_index_subblock_offset += shape.out_subblock_w;
                }

                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if constexpr (Activation::pack_relu) {
                if (last_out) {
                    PACK((llk_pack_relu_config(ReluConfig::none())));
                }
            }

            if constexpr (PackerL1Acc) {
                const bool keep_partials = last_out || (reload_last && block + 2 == shape.num_k_blocks);
                if (!keep_partials) {
                    for (uint32_t off = 0; off < out_block_num_tiles; off += out_subblock_num_tiles) {
                        interm_buf.wait_front(out_subblock_num_tiles);
                        interm_buf.pop_front(out_subblock_num_tiles);
                    }
                }
            }

            if (InputPolicy == matmul_config::InputPolicy::WaitAndPopPerKBlock || !last_out) {
                in0_buf.pop_front(in0_block_num_tiles);
                in1_buf.pop_front(in1_block_num_tiles);
            }
        }
        if constexpr (PackerL1Acc) {
            PACK((llk_pack_reconfig_l1_acc(0)));
        }
    }
}

}  // namespace compute_kernel_lib

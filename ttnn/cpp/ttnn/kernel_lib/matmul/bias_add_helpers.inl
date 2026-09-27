// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

template <BiasBroadcast Broadcast, typename PostBiasFn, typename Activation>
ALWI void add_bias_bcast_rows(
    uint32_t partials_cb_id,
    uint32_t bias_cb_id,
    uint32_t out_cb_id,
    BiasAddShape shape,
    PostBiasFn post_bias,
    uint32_t bias_offset) {
    DataflowBuffer partials_buf(partials_cb_id), out_buf(out_cb_id);

    const uint32_t in0_num_subblocks = shape.in0_num_subblocks;
    const uint32_t in1_num_subblocks = shape.in1_num_subblocks;
    const uint32_t out_subblock_h = shape.out_subblock_h;
    const uint32_t out_subblock_w = shape.out_subblock_w;

    const uint32_t out_num_tiles = out_subblock_h * out_subblock_w;

    ASSERT(in0_num_subblocks > 0);
    ASSERT(in1_num_subblocks > 0);
    ASSERT(out_subblock_h > 0);
    ASSERT(out_subblock_w > 0);
    ASSERT(out_num_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT);

    reconfig_data_format_srca(partials_cb_id);
    reconfig_data_format_srcb(bias_cb_id);
    pack_reconfig_data_format(out_cb_id);
    if constexpr (Activation::pack_relu) {
        pack_relu_config(ReluConfig::zero());
    }
    if constexpr (Broadcast == BiasBroadcast::RowBroadcast) {
        add_bcast_rows_init(partials_cb_id, bias_cb_id);
    } else {
        add_init(partials_cb_id, bias_cb_id);
    }

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
        int in1_index_subblock_offset = 0;
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
            partials_buf.wait_front(out_num_tiles);
            tile_regs_acquire();

            for (uint32_t i = 0, j = 0; j < out_subblock_h; j++) {
                uint32_t bias_tile_idx = bias_offset + in1_index_subblock_offset;
                for (uint32_t k = 0; k < out_subblock_w; k++, i++) {
                    if constexpr (Broadcast == BiasBroadcast::RowBroadcast) {
                        add_tiles_bcast_rows(partials_cb_id, bias_cb_id, i, bias_tile_idx, i);
                    } else {
                        add_tiles(partials_cb_id, bias_cb_id, i, bias_tile_idx, i);
                    }
                    bias_tile_idx++;
                }
            }

            post_bias(out_num_tiles);

            tile_regs_commit();
            partials_buf.pop_front(out_num_tiles);

            out_buf.reserve_back(out_num_tiles);
            if constexpr (Activation::activation != KernelActivation::NONE) {
                apply_activation_from_pack<
                    Activation::activation,
                    Activation::param0,
                    Activation::param1,
                    Activation::param2>(out_num_tiles);
            } else {
                tile_regs_wait();
            }
            for (uint32_t i = 0; i < out_num_tiles; i++) {
                pack_tile(i, out_cb_id);
            }
            tile_regs_release();
            out_buf.push_back(out_num_tiles);

            in1_index_subblock_offset += out_subblock_w;
        }
    }

    if constexpr (Activation::pack_relu) {
        pack_relu_config(ReluConfig::none());
    }
}

}  // namespace compute_kernel_lib

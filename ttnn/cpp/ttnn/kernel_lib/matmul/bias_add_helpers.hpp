// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/matmul/sfpu_activation_helpers.hpp"

namespace compute_kernel_lib {

/**
 * RowBroadcast adds one bias tile per output column across all M rows.
 * Elementwise adds bias tiles matching each output subblock.
 */
enum class BiasBroadcast { RowBroadcast, Elementwise };

// Subblock grid for bias addition.
struct BiasAddShape {
    uint32_t in0_num_subblocks;  // Subblock count along M.
    uint32_t in1_num_subblocks;  // Subblock count along N.
    uint32_t out_subblock_h;     // Subblock height in tiles.
    uint32_t out_subblock_w;     // Subblock width in tiles.

    static constexpr BiasAddShape of(
        uint32_t in0_num_subblocks, uint32_t in1_num_subblocks, uint32_t out_subblock_h, uint32_t out_subblock_w) {
        return {in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w};
    }
};

namespace bias_add_config {

// Called on the math thread after bias addition, before commit; receives the DST tile count.
struct NoPostBias {
    ALWI void operator()(uint32_t /* out_subblock_num_tiles */) const {}
};

}  // namespace bias_add_config

// Consume partials and publish output one subblock at a time.
// The caller owns bias synchronization and disables L1 accumulation before calling.
// PostBiasFn runs on math before commit; Activation runs on pack.
template <
    BiasBroadcast Broadcast = BiasBroadcast::RowBroadcast,
    typename PostBiasFn = bias_add_config::NoPostBias,
    typename Activation = NoneActivation>
ALWI void add_bias_bcast_rows(
    uint32_t partials_cb_id,
    uint32_t bias_cb_id,
    uint32_t out_cb_id,
    BiasAddShape shape,
    PostBiasFn post_bias = {},
    uint32_t bias_offset = 0);

}  // namespace compute_kernel_lib

#include "bias_add_helpers.inl"

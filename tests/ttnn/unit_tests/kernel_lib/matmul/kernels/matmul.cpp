// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul/matmul.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;
    constexpr uint32_t k_blocks = get_compile_time_arg_val(0);
    constexpr bool l1_acc = get_compile_time_arg_val(1);
    constexpr uint32_t post_op = get_compile_time_arg_val(2);  // See test_matmul_helpers.py for post-op cases.
    constexpr bool static_shape = get_compile_time_arg_val(3);
    constexpr uint32_t batches = get_compile_time_arg_val(4);
    constexpr bool same_cb = get_compile_time_arg_val(5);
    const auto shape = [] {
        if constexpr (static_shape) {
            return StaticMatmulShape<2, 2, 2, 2, 1, k_blocks, batches>{};
        } else {
            return MatmulShape::of(2, 2, 2, 2, 1, k_blocks, batches);
        }
    }();
    constexpr bool with_bias = post_op == 1 || post_op == 4 || post_op == 7;
    constexpr uint32_t in0 = 0, in1 = 1, partials = 2, bias = 3, out = 16;
    DataflowBuffer a(in0), b(in1), bias_buf(bias);
    constexpr uint32_t partials_cb_id = same_cb && !with_bias ? out : partials;
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out);
    matmul_block_init(in0, in1, false, 2, 2, 1);
    if constexpr (post_op == 3 || post_op == 4) {
        ActivationInitHelper<KernelActivation::RELU6>::init();
    }

    // Inputs reside in sharded L1 tensors, ordered by K block by the test.
    a.reserve_back(4 * k_blocks * batches);
    a.push_back(4 * k_blocks * batches);
    b.reserve_back(4 * k_blocks * batches);
    b.push_back(4 * k_blocks * batches);
    if constexpr (with_bias) {
        bias_buf.reserve_back(4);
        bias_buf.push_back(4);
    }
    using MatmulActivation = std::conditional_t<
        post_op == 2,
        ReluActivation,
        std::conditional_t<post_op == 3, Relu6Activation<>, NoneActivation>>;
    using BiasActivation = std::conditional_t<
        post_op == 7,
        ReluActivation,
        std::conditional_t<post_op == 4, Relu6Activation<>, NoneActivation>>;
    compute_kernel_lib::matmul<
        false,
        l1_acc,
        matmul_config::InitMode::Initialize,
        matmul_config::InputPolicy::WaitAndPopPerKBlock,
        matmul_config::DataFormatReconfig::InputAndOutput,
        MatmulActivation>(in0, in1, with_bias ? partials : out, partials_cb_id, shape);
    if constexpr (with_bias) {
        bias_buf.wait_front(4);
        pack_reconfig_data_format(out);
        add_bias_bcast_rows<BiasBroadcast::RowBroadcast, bias_add_config::NoPostBias, BiasActivation>(
            partials, bias, out, BiasAddShape::of(2, 2, 2, 2));
        bias_buf.pop_front(4);
    }
}

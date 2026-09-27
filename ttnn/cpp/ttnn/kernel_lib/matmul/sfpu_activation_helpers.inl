// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/eltwise_unary/hardtanh.h"
#include "api/compute/eltwise_unary/selu.h"
#include "api/compute/eltwise_unary/softplus.h"

namespace compute_kernel_lib {

namespace detail {

template <KernelActivation Act>
inline constexpr bool is_supported_activation =
    Act == KernelActivation::NONE || Act == KernelActivation::SILU || Act == KernelActivation::TANH ||
    Act == KernelActivation::GELU || Act == KernelActivation::RELU6 || Act == KernelActivation::SIGMOID ||
    Act == KernelActivation::HARDSIGMOID || Act == KernelActivation::HARDTANH || Act == KernelActivation::SELU ||
    Act == KernelActivation::SOFTPLUS;

}  // namespace detail

template <KernelActivation Act, uint32_t Param0, uint32_t Param1>
FORCE_INLINE void ActivationInitHelper<Act, Param0, Param1>::init() {
    static_assert(detail::is_supported_activation<Act>, "Unsupported KernelActivation type for fused activation init");

    if constexpr (Act == KernelActivation::SILU) {
        silu_tile_init_pack();
    } else if constexpr (Act == KernelActivation::TANH) {
        tanh_tile_init_pack<Param0 != 0>();
    } else if constexpr (Act == KernelActivation::GELU) {
        gelu_tile_init_pack<Param0 != 0>();
    } else if constexpr (Act == KernelActivation::RELU6) {
        relu_max_tile_init_pack();
    } else if constexpr (Act == KernelActivation::SIGMOID) {
        sigmoid_tile_init_pack<Param1 != 0>();
    } else if constexpr (Act == KernelActivation::HARDSIGMOID) {
        hardsigmoid_tile_init_pack();
    } else if constexpr (Act == KernelActivation::HARDTANH) {
        hardtanh_tile_init_pack();
    } else if constexpr (Act == KernelActivation::SELU) {
        selu_tile_init_pack();
    } else if constexpr (Act == KernelActivation::SOFTPLUS) {
        softplus_tile_init_pack();
    }
}

namespace detail {

template <KernelActivation Act, uint32_t Param0 = 0, uint32_t Param1 = 0, uint32_t Param2 = 0>
struct ActivationApplyHelper {
    static_assert(detail::is_supported_activation<Act>, "Unsupported KernelActivation type for fused activation apply");

    static_assert(
        Act != KernelActivation::SOFTPLUS || Param0 != 0,
        "SOFTPLUS Param0 (beta) must be non-zero to avoid division by zero");

    FORCE_INLINE static void apply(uint32_t tile_index) {
        if constexpr (Act == KernelActivation::SILU) {
            silu_tile_pack(tile_index);
        } else if constexpr (Act == KernelActivation::TANH) {
            tanh_tile_pack<Param0 != 0>(tile_index);
        } else if constexpr (Act == KernelActivation::GELU) {
            gelu_tile_pack<Param0 != 0>(tile_index);
        } else if constexpr (Act == KernelActivation::RELU6) {
            constexpr uint32_t max = (Param0 != 0) ? Param0 : 0x40c00000u;
            relu_max_tile_pack(tile_index, max);
        } else if constexpr (Act == KernelActivation::SIGMOID) {
            constexpr int vec_mode = (Param0 == 1) ? VectorMode::R : (Param0 == 2) ? VectorMode::C : VectorMode::RC;
            sigmoid_tile_pack<vec_mode, Param1 != 0>(tile_index);
        } else if constexpr (Act == KernelActivation::HARDSIGMOID) {
            hardsigmoid_tile_pack(tile_index);
        } else if constexpr (Act == KernelActivation::HARDTANH) {
            hardtanh_tile_pack(tile_index, Param0, Param1);
        } else if constexpr (Act == KernelActivation::SELU) {
            selu_tile_pack(tile_index, Param0, Param1);
        } else if constexpr (Act == KernelActivation::SOFTPLUS) {
            // The SFPU API takes beta reciprocal before threshold.
            softplus_tile_pack(tile_index, Param0, Param2, Param1);
        }
    }
};

}  // namespace detail

// Apply activation to DST on the packer thread, replacing tile_regs_wait().
template <KernelActivation Act, uint32_t Param0, uint32_t Param1, uint32_t Param2>
FORCE_INLINE void apply_activation_from_pack(uint32_t out_subblock_num_tiles) {
    PACK(TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));

    // Select the packer's DST half.
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
        detail::ActivationApplyHelper<Act, Param0, Param1, Param2>::apply(i);
    }

    // Wait for SFPU completion before packing.
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
}

}  // namespace compute_kernel_lib

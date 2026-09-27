// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"
#include "internal/risc_attribs.h"

using ttnn::operations::matmul::KernelActivation;

namespace compute_kernel_lib {

// Activation policy shared by matmul and bias helpers.
template <KernelActivation Act, uint32_t Param0 = 0, uint32_t Param1 = 0, uint32_t Param2 = 0, bool PackRelu = false>
struct ActivationOp {
    static constexpr KernelActivation activation = Act;
    static constexpr bool pack_relu = PackRelu;
    static constexpr uint32_t param0 = Param0;
    static constexpr uint32_t param1 = Param1;
    static constexpr uint32_t param2 = Param2;
};

using NoneActivation = ActivationOp<KernelActivation::NONE>;
// Hardware packer ReLU; no SFPU initialization or execution.
using ReluActivation = ActivationOp<KernelActivation::NONE, 0, 0, 0, true>;
using SiluActivation = ActivationOp<KernelActivation::SILU>;
using HardsigmoidActivation = ActivationOp<KernelActivation::HARDSIGMOID>;
// Fast=0 selects the accurate variant.
template <uint32_t Fast = 0>
using TanhActivation = ActivationOp<KernelActivation::TANH, Fast>;
template <uint32_t Fast = 0>
using GeluActivation = ActivationOp<KernelActivation::GELU, Fast>;
// MaxBits encodes a float; 0 selects 6.0f.
template <uint32_t MaxBits = 0>
using Relu6Activation = ActivationOp<KernelActivation::RELU6, MaxBits>;
// SigmoidActivation: VecMode 1=R, 2=C, else RC. Fast non-zero enables fast approximation.
template <uint32_t VecMode = 0, uint32_t Fast = 0>
using SigmoidActivation = ActivationOp<KernelActivation::SIGMOID, VecMode, Fast>;
// LowBits and HighBits encode floats.
template <uint32_t LowBits, uint32_t HighBits>
using HardtanhActivation = ActivationOp<KernelActivation::HARDTANH, LowBits, HighBits>;
// AlphaBits and LambdaBits encode floats.
template <uint32_t AlphaBits, uint32_t LambdaBits>
using SeluActivation = ActivationOp<KernelActivation::SELU, AlphaBits, LambdaBits>;
// Parameters encode floats; beta must be nonzero.
template <uint32_t BetaBits, uint32_t ThresholdBits, uint32_t BetaReciprocalBits>
using SoftplusActivation = ActivationOp<KernelActivation::SOFTPLUS, BetaBits, ThresholdBits, BetaReciprocalBits>;

// Initialize once at kernel startup; init and apply run on the packer thread (TRISC2).
template <KernelActivation Act, uint32_t Param0 = 0, uint32_t Param1 = 0>
struct ActivationInitHelper {
    FORCE_INLINE static void init();
};

// Apply activation to DST on the packer thread, replacing tile_regs_wait().
template <KernelActivation Act, uint32_t Param0 = 0, uint32_t Param1 = 0, uint32_t Param2 = 0>
FORCE_INLINE void apply_activation_from_pack(uint32_t out_subblock_num_tiles);

}  // namespace compute_kernel_lib

#include "sfpu_activation_helpers.inl"

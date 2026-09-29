// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <variant>

#include "internal/disaggregation/kv_layout_spec.hpp"

namespace tt::tt_metal::internal::disaggregation {
namespace {

// Skeleton-level tests: exercise the self-contained parts of the spec (TemporalPolicy,
// Distribution, MemLayout, AddressingMode) that do not require constructing a full TensorSpec.
// inner_footprint() and the TensorSpec-dependent behaviour are covered when they land.

constexpr uint32_t EXPECTED_WINDOW_TOKENS = 4096;
constexpr uint32_t EXPECTED_CONV_WIDTH = 3;  // conv kernel 4 -> retain last 3
constexpr uint32_t EXPECTED_MESH_AXIS = 1;

// --- TemporalPolicy ---

TEST(KvLayoutSpec, CPU_TemporalPolicyDefaultsToDense) {
    TemporalPolicy policy;  // std::variant default-constructs its first alternative
    EXPECT_TRUE(std::holds_alternative<temporal::Dense>(policy));
}

TEST(KvLayoutSpec, CPU_TemporalPolicyWindowCarriesWidth) {
    TemporalPolicy policy = temporal::Window{.width = WindowTokens{EXPECTED_WINDOW_TOKENS}};
    ASSERT_TRUE(std::holds_alternative<temporal::Window>(policy));
    EXPECT_EQ(std::get<temporal::Window>(policy).width.get(), EXPECTED_WINDOW_TOKENS);
}

TEST(KvLayoutSpec, CPU_TemporalPolicyRollingCarriesWidth) {
    TemporalPolicy policy = temporal::Rolling{.width = RollingWidth{EXPECTED_CONV_WIDTH}};
    ASSERT_TRUE(std::holds_alternative<temporal::Rolling>(policy));
    EXPECT_EQ(std::get<temporal::Rolling>(policy).width.get(), EXPECTED_CONV_WIDTH);
}

TEST(KvLayoutSpec, CPU_TemporalPolicyRecurrentIsNone) {
    // KDA / Mamba recurrent summary: positionless, addressed by checkpoint tag.
    TemporalPolicy policy = temporal::None{};
    EXPECT_TRUE(std::holds_alternative<temporal::None>(policy));
}

// --- Distribution ---

TEST(KvLayoutSpec, CPU_DistributionMixesShardAndReplicate) {
    // MLA-like: a head axis replicated, a sequence axis sharded onto a mesh axis.
    Distribution dist;
    dist.per_axis.push_back(Replicate{});
    dist.per_axis.push_back(Shard{.mesh_axis = MeshAxis{EXPECTED_MESH_AXIS}});

    ASSERT_EQ(dist.per_axis.size(), 2u);
    EXPECT_TRUE(std::holds_alternative<Replicate>(dist.per_axis[0]));
    ASSERT_TRUE(std::holds_alternative<Shard>(dist.per_axis[1]));
    EXPECT_EQ(std::get<Shard>(dist.per_axis[1]).mesh_axis.get(), EXPECTED_MESH_AXIS);
}

// --- MemLayout / AddressingMode defaults ---

TEST(KvLayoutSpec, CPU_MemLayoutDefaultsToOptimalBankOrder) {
    MemLayout layout;
    EXPECT_EQ(layout.bank_order, BankOrder::Optimal);
}

TEST(KvLayoutSpec, CPU_AddressingSupportsBothSlotAndPaged) {
    // Both representations are first-class; the spec carries which one applies per tensor.
    EXPECT_NE(AddressingMode::Slot, AddressingMode::Paged);
}

}  // namespace
}  // namespace tt::tt_metal::internal::disaggregation

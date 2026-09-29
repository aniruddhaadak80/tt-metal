// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include <tt-metalium/shape.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>

#include "internal/disaggregation/kv_layout_spec.hpp"
#include "internal/disaggregation/noc_addr.hpp"
#include "internal/disaggregation/to_chunk_map.hpp"

namespace tt::tt_metal::internal::disaggregation {
namespace {

// These tests drive the ONE `to_chunk_map` factory with REAL tt-metal TensorSpecs and compare the
// resulting KvChunkAddressTable against a C++ port of blaze's migration-path arithmetic (the same
// reference the device-free kv_layout_spec_smoke scripts use). CPU-only; no device required.

constexpr uint32_t kTileLocal = 32;
constexpr uint32_t kBfp8TileBytesLocal = 1088;

// Build a real bfloat8_b, TILE-layout, DRAM-interleaved TensorSpec of the given logical shape.
TensorSpec make_bfp8_dram_spec(const Shape& shape) {
    auto page_config = PageConfig(Layout::TILE);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::BFLOAT8_B, page_config, memory_config);
    return TensorSpec(shape, tensor_layout);
}

uint32_t bfp8_chunk_bytes(uint32_t tokens_per_chunk, uint64_t feature_dim) {
    return (tokens_per_chunk / kTileLocal) * static_cast<uint32_t>(feature_dim / kTileLocal) * kBfp8TileBytesLocal;
}

// --- Reference: blaze MLA migration arithmetic (deepseek_v3_kimi_k2_mla.py) ---
struct RefLoc {
    uint32_t bank_id;
    uint64_t offset;
    uint32_t owner;  // sp row
};

RefLoc mla_reference(
    uint32_t position,
    uint32_t slot,
    uint64_t base,
    uint64_t head_dim,
    uint32_t per_device_seq_len,
    uint32_t tokens_per_chunk,
    uint32_t sp_dim,
    uint32_t k_chunk_size,
    uint32_t device_chunk_size,
    uint32_t sp_origin) {
    static constexpr std::array<uint32_t, 8> kOrder = {1, 3, 2, 0, 5, 7, 6, 4};
    const uint32_t num_banks = 8;
    const uint32_t round_idx = position / (device_chunk_size * sp_dim);
    const uint32_t in_round = position % (device_chunk_size * sp_dim);
    const uint32_t sp_device_idx = in_round / device_chunk_size;
    const uint32_t in_chunk_dev = in_round % device_chunk_size;
    const uint32_t owner = (sp_device_idx + sp_origin) % sp_dim;
    const uint32_t local_pos = round_idx * device_chunk_size + in_chunk_dev;

    const uint32_t csb = bfp8_chunk_bytes(tokens_per_chunk, head_dim);
    const uint32_t chunks_per_slot = per_device_seq_len / k_chunk_size;
    const uint32_t local_chunk = local_pos / k_chunk_size;
    const uint32_t in_chunk = local_pos % k_chunk_size;
    const uint32_t shard_id = slot * chunks_per_slot + local_chunk;
    const uint32_t shard_size_b = (k_chunk_size / tokens_per_chunk) * csb;
    const uint32_t bank_id = kOrder[shard_id % num_banks];
    const uint64_t noc = base + static_cast<uint64_t>(shard_id / num_banks) * shard_size_b + (in_chunk / tokens_per_chunk) * csb;
    return {bank_id, noc, owner};
}

// --- Reference: gpt-oss GQA chunk_location (gpt_oss.py) ---
RefLoc gptoss_reference(
    uint32_t position,
    uint32_t slot,
    uint64_t base,
    uint32_t local_head,
    uint32_t bph,
    uint32_t st_pb,
    uint32_t row_bytes,
    bool cyclic,
    uint32_t sk_chunk_t) {
    static constexpr std::array<uint32_t, 8> kOrder = {1, 3, 2, 0, 5, 7, 6, 4};
    const uint32_t tile_row = position / kTileLocal;
    uint32_t bank_slice = 0;
    uint32_t within = 0;
    if (cyclic) {
        const uint32_t chunk = tile_row / sk_chunk_t;
        bank_slice = chunk % bph;
        within = (chunk / bph) * sk_chunk_t + (tile_row % sk_chunk_t);
    } else {
        bank_slice = tile_row / st_pb;
        within = tile_row % st_pb;
    }
    const uint32_t bank_id = kOrder[local_head * bph + bank_slice];
    const uint64_t noc = base + static_cast<uint64_t>(slot) * st_pb * row_bytes + static_cast<uint64_t>(within) * row_bytes;
    return {bank_id, noc, 0};
}

// --- MLA (DeepSeek-V3 / Kimi) ---
TEST(ToChunkMap, CPU_MlaShardMatchesMigrationReference) {
    const uint64_t F = 576;  // kv_lora_rank 512 + qk_rope 64
    const uint32_t sp_dim = 4;
    const uint32_t mesh_cols = 2;
    const uint32_t k_chunk = 128;
    const uint32_t chunk_n_tokens = kTileLocal;
    const uint32_t device_chunk_size = k_chunk * 8;  // 1024 (DeepSeek default)
    const uint32_t max_seq_len = device_chunk_size * sp_dim * 2;  // 8192, multiple of dcs*sp_dim
    const uint32_t num_slots = 3;
    const uint64_t base = 0x1000'0000ull;
    const uint32_t per_dev = max_seq_len / sp_dim;

    // Real TensorSpec: (slot, F, seq), seq axis = 2. Round feature/seq to tile multiples.
    KvLayoutSpec spec{.tensor = make_bfp8_dram_spec(Shape{num_slots, static_cast<uint32_t>(F), max_seq_len})};
    spec.seq_axis = SeqAxis{2};
    spec.temporal = temporal::Dense{};
    spec.distribution.per_axis = {Replicate{}, Replicate{}, Shard{.mesh_axis = MeshAxis{0}}};
    spec.mem_layout.bank_order = BankOrder::Optimal;
    spec.mem_layout.bank_scheme = BankScheme::MlaShard;
    spec.mem_layout.k_chunk_size = KChunkSize{k_chunk};
    spec.mem_layout.chunk_n_tokens = ChunkTokens{chunk_n_tokens};
    spec.addressing = AddressingMode::Slot;
    spec.sp_dim = SpDim{sp_dim};
    spec.mesh_cols = MeshCols{mesh_cols};
    spec.device_chunk_size = DeviceChunkSize{device_chunk_size};

    // Verify inner_footprint excludes the batch/slot axis (axis 0) and the seq axis -> F.
    EXPECT_EQ(spec.inner_footprint().get(), F);

    MapGeometry geom;
    geom.num_layers = 1;
    geom.num_slots = num_slots;
    geom.max_seq_len = max_seq_len;
    geom.position_step = chunk_n_tokens;
    geom.base_addr = base;

    auto table = to_chunk_map({spec}, geom);

    uint32_t checked = 0;
    for (uint32_t slot = 0; slot < num_slots; ++slot) {
        for (uint32_t pos = 0; pos < max_seq_len; pos += chunk_n_tokens) {
            const RefLoc ref = mla_reference(
                pos, slot, base, F, per_dev, chunk_n_tokens, sp_dim, k_chunk, device_chunk_size, 0);
            const KvCacheLocation loc = table.lookup(0, pos, slot, 0u);
            EXPECT_EQ(addr_channel(loc.noc_addr), ref.bank_id) << "pos=" << pos << " slot=" << slot;
            EXPECT_EQ(addr_local(loc.noc_addr), static_cast<uint32_t>(ref.offset & 0xFFFFFFFFull))
                << "pos=" << pos << " slot=" << slot;

            // Device group: MLA replicates the owning sp row across all mesh columns.
            const auto& grp = table.get_device_group(loc.device_group_index);
            ASSERT_EQ(grp.fabric_node_ids.size(), mesh_cols);
            for (uint32_t c = 0; c < mesh_cols; ++c) {
                EXPECT_EQ(grp.fabric_node_ids[c].chip_id, ref.owner * mesh_cols + c);
            }
            ++checked;
        }
    }
    EXPECT_GT(checked, 0u);
}

// --- gpt-oss GQA (CYCLIC dense layer) ---
TEST(ToChunkMap, CPU_GqaCyclicMatchesReference) {
    const uint32_t n_kv_heads = 8;
    const uint32_t tp = 8;
    const uint64_t head_dim = 64;
    const uint32_t sdpa_k_chunk = 512;
    const uint32_t max_seq_len = 4096;
    const uint32_t num_slots = 2;
    const uint64_t base = 0x10000ull;
    const uint32_t n_kv_heads_per_dev = std::max(1u, n_kv_heads / tp);  // 1
    const uint32_t bph = 8;  // banks_per_kv_head for head_dim 64, 1 head/dev
    const uint32_t row_bytes = static_cast<uint32_t>((head_dim / kTileLocal) * kBfp8TileBytesLocal);
    const uint32_t sk_chunk_t = sdpa_k_chunk / kTileLocal;
    const uint32_t st_pb = (max_seq_len / kTileLocal) / bph;

    KvLayoutSpec spec{.tensor = make_bfp8_dram_spec(
                          Shape{num_slots, n_kv_heads, max_seq_len, static_cast<uint32_t>(head_dim)})};
    spec.seq_axis = SeqAxis{2};
    spec.temporal = temporal::Dense{};
    spec.distribution.per_axis = {Replicate{}, Shard{.mesh_axis = MeshAxis{1}}, Replicate{}, Replicate{}};
    spec.mem_layout.bank_order = BankOrder::Optimal;
    spec.mem_layout.bank_scheme = BankScheme::Cyclic;
    spec.mem_layout.k_chunk_size = KChunkSize{sdpa_k_chunk};
    spec.mem_layout.chunk_n_tokens = ChunkTokens{kTileLocal};
    spec.mem_layout.banks_per_head = BanksPerHead{bph};
    spec.addressing = AddressingMode::Slot;
    spec.sp_dim = SpDim{1};
    spec.mesh_cols = MeshCols{n_kv_heads};
    spec.head_shard_axis = HeadShardAxis{1};

    // inner_footprint excludes batch axis (0), head-shard axis (1) and seq axis (2) -> head_dim.
    EXPECT_EQ(spec.inner_footprint().get(), head_dim);

    MapGeometry geom;
    geom.num_layers = 1;
    geom.num_slots = num_slots;
    geom.max_seq_len = max_seq_len;
    geom.position_step = kTileLocal;
    geom.base_addr = base;

    auto table = to_chunk_map({spec}, geom);

    const bool cyclic = true;
    uint32_t checked = 0;
    for (uint32_t head = 0; head < n_kv_heads; ++head) {
        const uint32_t local_head = head % n_kv_heads_per_dev;
        const uint32_t chip = head / n_kv_heads_per_dev;
        for (uint32_t slot = 0; slot < num_slots; ++slot) {
            const uint32_t map_slot = head * num_slots + slot;
            for (uint32_t pos = 0; pos < max_seq_len; pos += kTileLocal) {
                const RefLoc ref =
                    gptoss_reference(pos, slot, base, local_head, bph, st_pb, row_bytes, cyclic, sk_chunk_t);
                const KvCacheLocation loc = table.lookup(0, pos, map_slot, 0u);
                EXPECT_EQ(addr_channel(loc.noc_addr), ref.bank_id) << "head=" << head << " pos=" << pos;
                EXPECT_EQ(addr_local(loc.noc_addr), static_cast<uint32_t>(ref.offset & 0xFFFFFFFFull))
                    << "head=" << head << " pos=" << pos;
                const auto& grp = table.get_device_group(loc.device_group_index);
                ASSERT_EQ(grp.fabric_node_ids.size(), 1u);
                EXPECT_EQ(grp.fabric_node_ids[0].chip_id, chip);
                ++checked;
            }
        }
    }
    EXPECT_GT(checked, 0u);
}

}  // namespace
}  // namespace tt::tt_metal::internal::disaggregation

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/to_chunk_map.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include <internal/disaggregation/noc_addr.hpp>

namespace tt::tt_metal::internal::disaggregation {

namespace {

using tt::tt_fabric::FabricNodeId;

// A single resolved chunk: physical bank + per-bank offset + the mesh (row, col) coordinates that
// hold its replicas. Mirrors the reference factory's (bank_id, per_bank_offset, device_coords).
struct Located {
    uint32_t bank_id = 0;
    uint64_t offset = 0;
    std::vector<std::pair<uint32_t, uint32_t>> coords;  // (row, col) mesh coordinates
};

// The active bank ordering: the OPTIMAL permutation, or the identity round-robin.
std::array<uint32_t, kNumDramBanks> bank_table(const MemLayout& mem) {
    std::array<uint32_t, kNumDramBanks> banks{};
    if (mem.bank_order == BankOrder::Optimal) {
        banks = kOptimalDramBankOrder;
    } else {
        for (uint32_t i = 0; i < kNumDramBanks; ++i) {
            banks[i] = i;
        }
    }
    return banks;
}

// MLA CP ownership: round-robin over sp devices with a per-device chunk stride.
std::pair<uint32_t, uint32_t> cp_mla_stride(const KvLayoutSpec& spec, uint32_t position, uint32_t dcs) {
    const uint32_t sp_dim = spec.sp_dim.get();
    const uint32_t round_idx = position / (dcs * sp_dim);
    const uint32_t in_round = position % (dcs * sp_dim);
    const uint32_t sp_device_idx = in_round / dcs;
    const uint32_t in_chunk = in_round % dcs;
    const uint32_t owner = (sp_device_idx + spec.sp_origin.get()) % sp_dim;
    return {owner, round_idx * dcs + in_chunk};
}

// GLM CP ownership: owner = chunk_index % num_devices.
std::pair<uint32_t, uint32_t> cp_chunk_modulo(const KvLayoutSpec& spec, uint32_t position, uint32_t dcs) {
    const uint32_t sp_dim = spec.sp_dim.get();
    if (sp_dim <= 1 || dcs == 0) {
        return {0, position};
    }
    const uint32_t chunk_index = position / dcs;
    const uint32_t owner = (chunk_index + spec.sp_origin.get()) % sp_dim;
    const uint32_t local_pos = (chunk_index / sp_dim) * dcs + (position % dcs);
    return {owner, local_pos};
}

Located locate_one(
    const KvLayoutSpec& spec,
    uint32_t layer,
    std::optional<uint32_t> head,
    uint32_t position,
    uint32_t slot,
    const MapGeometry& geom) {
    const MemLayout& mem = spec.mem_layout;
    const uint32_t tpc = mem.chunk_n_tokens.get();
    const auto kcs_it = spec.k_chunk_by_layer.find(layer);
    const uint32_t kcs = kcs_it != spec.k_chunk_by_layer.end() ? kcs_it->second.get() : mem.k_chunk_size.get();
    const auto scheme_it = spec.bank_scheme_by_layer.find(layer);
    const BankScheme scheme = scheme_it != spec.bank_scheme_by_layer.end() ? scheme_it->second : mem.bank_scheme;
    const auto banks = bank_table(mem);

    const uint64_t f = spec.inner_footprint().get();
    const uint32_t sp_dim = spec.sp_dim.get();
    const uint32_t mesh_cols = spec.mesh_cols.get();
    const uint32_t per_dev_seq = geom.max_seq_len / sp_dim;
    const uint64_t base = geom.base_addr;

    uint32_t dcs =
        spec.device_chunk_size.has_value() ? spec.device_chunk_size->get() : kcs * mem.num_banks;

    const auto& shape = spec.tensor.logical_shape();

    // ---- MLA_SHARD: CP stride + shard_id stacking + OPTIMAL perm over num_banks ----
    if (scheme == BankScheme::MlaShard) {
        const uint32_t csb = chunk_size_bytes(mem, tpc, f);
        const auto [owner, local_pos] = cp_mla_stride(spec, position, dcs);
        const uint32_t chunks_per_slot = per_dev_seq / kcs;
        const uint32_t local_chunk = local_pos / kcs;
        const uint32_t in_chunk = local_pos % kcs;
        const uint32_t shard_id =
            spec.addressing == AddressingMode::Slot ? (slot * chunks_per_slot + local_chunk) : local_chunk;
        const uint32_t shard_size_b = (kcs / tpc) * csb;
        const uint32_t bank_id = banks[shard_id % mem.num_banks];
        const uint64_t off =
            base + static_cast<uint64_t>(shard_id / mem.num_banks) * shard_size_b + (in_chunk / tpc) * csb;
        std::vector<std::pair<uint32_t, uint32_t>> coords;
        for (uint32_t c = 0; c < mesh_cols; ++c) {
            coords.emplace_back(owner, c);
        }
        return {bank_id, off, std::move(coords)};
    }

    // ---- BLOCK_CYCLIC: minimax index_k, column-split by idx_cp, round-robin blocks over banks ----
    if (scheme == BankScheme::BlockCyclic) {
        const uint32_t idx_cp = spec.idx_cp.get();
        const uint32_t block = kcs;
        const uint32_t idx_row_bytes = static_cast<uint32_t>((f / kTile) * kBfp8TileBytes);
        const uint32_t n_blocks = per_dev_seq / block;
        const uint32_t n_blocks_dev = (n_blocks + idx_cp - 1) / idx_cp;
        const uint32_t num_banks = std::min(mem.banks_per_head.get(), n_blocks_dev);
        const uint32_t blocks_per_bank = n_blocks_dev / num_banks;
        const uint32_t gb = position / block;
        const uint32_t dev = gb % idx_cp;
        const uint32_t local_position = (gb / idx_cp) * block + position % block;
        const uint32_t global_block = slot * blocks_per_bank * num_banks + local_position / block;
        const uint32_t bank_id = banks[global_block % num_banks];
        const uint32_t local_block = global_block / num_banks;
        const uint32_t tile_row = (local_block * block + (local_position % block)) / kTile;
        const uint64_t off = base + static_cast<uint64_t>(tile_row) * idx_row_bytes;
        std::vector<std::pair<uint32_t, uint32_t>> coords;
        for (uint32_t r = 0; r < spec.mesh_rows.get(); ++r) {
            for (uint32_t c = 0; c < mesh_cols; ++c) {
                if (c % idx_cp == dev) {
                    coords.emplace_back(0u, r * mesh_cols + c);
                }
            }
        }
        return {bank_id, off, std::move(coords)};
    }

    // ---- BLOCK with GQA group: minimax K/V height-sharded, group -> mesh row block ----
    if (scheme == BankScheme::Block && spec.group.has_value()) {
        const uint32_t bph = mem.banks_per_head.get();
        const uint32_t row_bytes = static_cast<uint32_t>((f / kTile) * kBfp8TileBytes);
        const uint32_t st_pb = (per_dev_seq / kTile) / bph;
        const uint32_t chunk = position / kcs;
        const uint32_t off_in_chunk = position % kcs;
        const uint32_t bank_slice = chunk % bph;
        const uint32_t within = (chunk / bph) * (kcs / kTile) + off_in_chunk / kTile;
        const uint32_t bank_id = banks[bank_slice];
        const uint64_t off = base + static_cast<uint64_t>(slot) * st_pb * row_bytes + static_cast<uint64_t>(within) * row_bytes;
        std::vector<std::pair<uint32_t, uint32_t>> coords;
        for (uint32_t c = 0; c < mesh_cols; ++c) {
            coords.emplace_back(0u, spec.group->get() * mesh_cols + c);
        }
        return {bank_id, off, std::move(coords)};
    }

    // ---- BLOCK / CYCLIC per-head GQA: head -> TP shard (gpt-oss) ----
    if ((scheme == BankScheme::Block || scheme == BankScheme::Cyclic) && spec.head_shard_axis.has_value()) {
        TT_FATAL(head.has_value(), "per-head scheme requires an enumerated head");
        const uint32_t head_axis = spec.head_shard_axis->get();
        const uint32_t n_heads = shape[head_axis];
        const uint32_t n_heads_per_dev = std::max(1u, n_heads / (mesh_cols * std::max(1u, sp_dim)));
        const uint32_t bph = mem.banks_per_head.get();
        const uint64_t dht = f / kTile;
        const uint32_t row_bytes = static_cast<uint32_t>(dht * kBfp8TileBytes);
        const uint32_t chip = *head / n_heads_per_dev;
        const uint32_t local_head = *head % n_heads_per_dev;
        const uint32_t seq_extent = shape[spec.seq_axis->get()];
        const uint32_t st_pb = (seq_extent / kTile) / bph;
        const uint32_t sk_chunk_t = kcs / kTile;
        const uint32_t tile_row = position / kTile;
        uint32_t bank_slice = 0;
        uint32_t within = 0;
        if (scheme == BankScheme::Cyclic) {
            const uint32_t chunk = tile_row / sk_chunk_t;
            bank_slice = chunk % bph;
            within = (chunk / bph) * sk_chunk_t + (tile_row % sk_chunk_t);
        } else {
            bank_slice = tile_row / st_pb;
            within = tile_row % st_pb;
        }
        const uint32_t bank_id = banks[local_head * bph + bank_slice];
        const uint64_t off = base + static_cast<uint64_t>(slot) * st_pb * row_bytes + static_cast<uint64_t>(within) * row_bytes;
        std::vector<std::pair<uint32_t, uint32_t>> coords = {{spec.sp_origin.get(), chip}};
        return {bank_id, off, std::move(coords)};
    }

    // ---- NATURAL: GLM-style paged CP; IDENTITY (page%banks) vs OPTIMAL-perm (perm[chunk%blocks]) ----
    const auto [owner, local_pos] = cp_chunk_modulo(spec, position, dcs);
    const uint32_t csb = chunk_size_bytes(mem, tpc, f);
    const uint32_t tpc_page = kcs;
    if (mem.bank_order == BankOrder::Identity) {
        const uint32_t pages_per_slot = geom.num_slots > 1 ? (per_dev_seq / tpc_page) : 0;
        const uint32_t eff_slot = pages_per_slot > 0 ? slot : 0;
        const uint32_t page_id = eff_slot * pages_per_slot + local_pos / tpc_page;
        const uint32_t in_page_offset = (local_pos % tpc_page) * (csb / tpc_page);
        const uint32_t bank_id = page_id % mem.num_banks;
        const uint64_t off = base + static_cast<uint64_t>(page_id / mem.num_banks) * csb + in_page_offset;
        return {bank_id, off, {{owner, 0u}}};
    }
    // OPTIMAL indexer: fixed permutation over num_blocks; per-bank slot stacking (ND-shard).
    const uint32_t nblk = mem.num_blocks;
    const uint32_t chunk_idx = local_pos / tpc_page;
    const uint32_t bank_id = kOptimalDramBankOrder[chunk_idx % nblk];
    const uint32_t within_bank = chunk_idx / nblk;
    const uint32_t slot_size_b = (per_dev_seq / tpc_page * csb) / nblk;
    const uint32_t eff_slot = geom.num_slots > 1 ? slot : 0;
    const uint64_t off = base + static_cast<uint64_t>(within_bank) * csb + static_cast<uint64_t>(slot_size_b) * eff_slot;
    return {bank_id, off, {{owner, 0u}}};
}

// (row, col) mesh coordinate -> logical chip id (row-major within the mesh).
FabricNodeId to_fabric_node(tt::tt_fabric::MeshId mesh_id, uint32_t row, uint32_t col, uint32_t mesh_cols) {
    return FabricNodeId(mesh_id, row * std::max(1u, mesh_cols) + col);
}

}  // namespace

KvChunkAddressTable to_chunk_map(const std::vector<KvLayoutSpec>& specs, const MapGeometry& geometry) {
    TT_FATAL(!specs.empty(), "to_chunk_map requires at least one spec");

    // One config per spec. A per-head spec fans heads into extra slots: num_slots *= n_heads.
    std::vector<KvChunkAddressTableConfig> configs;
    std::vector<uint32_t> heads_per_config;
    for (const auto& spec : specs) {
        const uint32_t n_heads =
            spec.head_shard_axis.has_value() ? spec.tensor.logical_shape()[spec.head_shard_axis->get()] : 1;
        heads_per_config.push_back(n_heads);
        configs.push_back(KvChunkAddressTableConfig{
            .num_layers = geometry.num_layers,
            .max_sequence_length = geometry.max_seq_len,
            .num_slots = geometry.num_slots * n_heads,
            .chunk_n_tokens = spec.mem_layout.chunk_n_tokens.get(),
            .chunk_size_bytes = chunk_size_bytes(
                spec.mem_layout, spec.mem_layout.chunk_n_tokens.get(), spec.inner_footprint().get()),
        });
    }

    KvChunkAddressTable table{std::span<const KvChunkAddressTableConfig>(configs)};

    for (uint32_t cfg = 0; cfg < specs.size(); ++cfg) {
        const KvLayoutSpec& spec = specs[cfg];
        const uint32_t n_heads = heads_per_config[cfg];
        const bool per_head = spec.head_shard_axis.has_value();

        for (uint32_t layer = 0; layer < geometry.num_layers; ++layer) {
            if (!spec.applies_to_layer(layer)) {
                continue;
            }
            const auto tp_it = spec.temporal_by_layer.find(layer);
            const TemporalPolicy& temporal = tp_it != spec.temporal_by_layer.end() ? tp_it->second : spec.temporal;
            uint32_t extent = geometry.max_seq_len;
            if (const auto* w = std::get_if<temporal::Window>(&temporal)) {
                if (w->width.get() != 0) {
                    extent = std::min(extent, w->width.get());
                }
            }

            for (uint32_t h = 0; h < n_heads; ++h) {
                const std::optional<uint32_t> head = per_head ? std::optional<uint32_t>(h) : std::nullopt;
                for (uint32_t slot = 0; slot < geometry.num_slots; ++slot) {
                    for (uint32_t pos = 0; pos < extent && pos < geometry.max_seq_len; pos += geometry.position_step) {
                        const Located loc = locate_one(spec, layer, head, pos, slot, geometry);
                        std::vector<FabricNodeId> nodes;
                        nodes.reserve(loc.coords.size());
                        for (const auto& [row, col] : loc.coords) {
                            nodes.push_back(to_fabric_node(geometry.mesh_id, row, col, spec.mesh_cols.get()));
                        }
                        const DeviceGroupIndex grp = table.add_device_group(std::move(nodes));
                        const uint64_t noc_addr = (static_cast<uint64_t>(loc.bank_id) << 32) | addr_local(loc.offset);
                        const uint32_t map_slot = per_head ? (h * geometry.num_slots + slot) : slot;
                        table.set(
                            layer,
                            pos,
                            map_slot,
                            KvCacheLocation{
                                .noc_addr = noc_addr,
                                .size_bytes = configs[cfg].chunk_size_bytes,
                                .device_group_index = grp},
                            cfg);
                    }
                }
            }
        }
    }

    return table;
}

}  // namespace tt::tt_metal::internal::disaggregation

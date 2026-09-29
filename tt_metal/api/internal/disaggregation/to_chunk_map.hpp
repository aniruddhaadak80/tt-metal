// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>

#include <internal/disaggregation/kv_chunk_address_table.hpp>
#include <internal/disaggregation/kv_layout_spec.hpp>

namespace tt::tt_metal::internal::disaggregation {

// Runtime extents the chunk map is enumerated over. Mirrors the device-free reference factory's
// `Geometry` (kv_manager/tests/kv_layout_spec_smoke/kv_layout_spec.py): the (layer, slot, position)
// grid plus the DRAM base address the offsets are relative to.
struct MapGeometry {
    uint32_t num_layers = 0;
    uint32_t num_slots = 0;
    uint32_t max_seq_len = 0;         // in tokens
    uint32_t position_step = kTile;   // token stride between enumerated positions
    uint64_t base_addr = 0;
    tt::tt_fabric::MeshId mesh_id{0};  // fabric mesh the device coords resolve into
};

// The commonized factory. `specs` is a model expressed as an ordered LIST of co-resident
// KvLayoutSpecs (K/V + indexer, per-layer applicability); a single spec is a one-element model.
// ONE dispatch over bank_scheme x temporal x distribution fills a KvChunkAddressTable: each
// (config, layer, slot, position[, head]) resolves to a KvCacheLocation whose noc_addr encodes
// (bank_id << 32) | per_bank_offset (matching noc_addr.hpp) and whose device_group_index references
// the replica set of mesh coordinates the placement implies.
//
// One config per spec; a per-head spec (BLOCK/CYCLIC GQA) fans its heads out into ADDITIONAL slots
// so heads and slots share one flat slot axis: config slot = head * num_slots + slot.
KvChunkAddressTable to_chunk_map(const std::vector<KvLayoutSpec>& specs, const MapGeometry& geometry);

}  // namespace tt::tt_metal::internal::disaggregation

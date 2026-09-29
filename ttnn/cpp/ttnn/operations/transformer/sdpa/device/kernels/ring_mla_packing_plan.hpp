// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Shared host/device contract. All extents are tiles. Source IDs are tensor ranks,
// never transport ranks; the caller translates the resolved route before packing.
// Live source width controls packing, while allocated source width controls addresses.
// A partial K chunk retains the configured full CB stride in every consumer.

// One contiguous sequence interval in a packed K chunk. Column starts are implicit:
// zero for the first run, then the preceding run's exclusive column_end.
struct PackedKVMaskRun {
    uint32_t global_start_tile;
    uint32_t column_end;
};

// Preconditions checked by the host: nonzero source/chunk sizes, group size <= 32,
// products representable in uint32_t, source IDs in the resolved tensor-rank range.
// Chunk and stream indices passed to addressing helpers must be in range.
// A packed attention pass concatenates equally sized physical KV sources. All sizes
// are in tiles; the final chunk may be partial, but retains the full CB stride.
struct PackedKVGroupPlan {
    uint32_t source_tiles;
    uint32_t source_count;
    uint32_t chunk_tiles;

    constexpr uint32_t tile_count() const { return source_tiles * source_count; }
    constexpr uint32_t chunk_count() const { return tile_count() / chunk_tiles + (tile_count() % chunk_tiles != 0); }
    constexpr uint32_t valid_tiles(uint32_t chunk) const {
        const uint32_t start = chunk * chunk_tiles;
        if (start >= tile_count()) {
            return 0;
        }
        const uint32_t remaining = tile_count() - start;
        return remaining < chunk_tiles ? remaining : chunk_tiles;
    }
    constexpr uint32_t source_index(uint32_t stream_tile) const { return stream_tile / source_tiles; }
    constexpr uint32_t source_offset(uint32_t stream_tile) const { return stream_tile % source_tiles; }
    constexpr uint32_t last_source(uint32_t chunk) const {
        return source_index(chunk * chunk_tiles + valid_tiles(chunk) - 1);
    }
    constexpr uint32_t segment_tiles(uint32_t chunk, uint32_t destination_offset) const {
        const uint32_t remaining = valid_tiles(chunk) - destination_offset;
        const uint32_t source_remaining = source_tiles - source_offset(chunk * chunk_tiles + destination_offset);
        return remaining < source_remaining ? remaining : source_remaining;
    }
    // Largest local slab index among the chunk's valid tiles. A chunk that crosses into a later
    // source contains the tail of the earlier one, i.e. that source's last slab.
    constexpr uint32_t max_slab(uint32_t chunk, uint32_t region_tiles) const {
        const uint32_t first = chunk * chunk_tiles;
        const uint32_t last = first + valid_tiles(chunk) - 1;
        if (source_index(first) != source_index(last)) {
            return (source_tiles - 1) / region_tiles;
        }
        return source_offset(last) / region_tiles;
    }
    // Emit at most chunk_tiles runs, splitting at both source and block-cyclic slab
    // boundaries. The caller supplies chunk_tiles entries, including for partial chunks.
    // Runs once per (Q chunk, K chunk) on every compute thread, so it walks the chunk with
    // running source/region cursors: one division by the runtime source width per call.
    constexpr uint32_t mask_runs(
        uint32_t chunk,
        const uint32_t* source_ids,
        uint32_t region_tiles,
        uint32_t global_chunk_tiles,
        PackedKVMaskRun* runs) const {
        const uint32_t valid = valid_tiles(chunk);
        const uint32_t stream_start = chunk * chunk_tiles;
        uint32_t source = source_index(stream_start);
        uint32_t local = stream_start - source * source_tiles;
        uint32_t slab = local / region_tiles;
        uint32_t region_offset = local - slab * region_tiles;
        uint32_t count = 0;
        for (uint32_t column = 0; column < valid;) {
            const uint32_t source_remaining = source_tiles - local;
            const uint32_t region_remaining = region_tiles - region_offset;
            uint32_t length = valid - column;
            length = length < source_remaining ? length : source_remaining;
            length = length < region_remaining ? length : region_remaining;
            runs[count++] = {
                slab * global_chunk_tiles + source_ids[source] * region_tiles + region_offset, column + length};
            column += length;
            local += length;
            region_offset += length;
            if (local == source_tiles) {
                ++source;
                local = 0;
                slab = 0;
                region_offset = 0;
            } else if (region_offset == region_tiles) {
                ++slab;
                region_offset = 0;
            }
        }
        return count;
    }
};

// Readiness advances once per source in sequencer order, and is reused across Q
// chunks in the same group. The callback must complete its wait before returning.
// Drain also covers cores with no Q work, preserving the receiver's signal cadence.
struct PackedKVSourceReadiness {
    uint32_t ready_sources = 0;

    template <typename WaitSource>
    void drain(uint32_t required_sources, WaitSource&& wait_source) {
        while (ready_sources < required_sources) {
            wait_source(ready_sources);
            ++ready_sources;
        }
    }

    template <typename WaitSource>
    void wait_for_chunk(const PackedKVGroupPlan& plan, uint32_t chunk, WaitSource&& wait_source) {
        drain(plan.last_source(chunk) + 1, wait_source);
    }
};

// Each source owns one region per global prefill chunk. Clip an oversized
// input cache to the slabs touched by this invocation; keep the physical source
// stride separate so remote reads still address the persistent gather correctly.
constexpr uint32_t packed_kv_source_tiles(
    uint32_t capacity_tiles, uint32_t logical_tiles, uint32_t region_tiles, uint32_t ring_size) {
    if (region_tiles == 0 || ring_size == 0) {
        return capacity_tiles;
    }
    const uint32_t global_chunk_tiles = region_tiles * ring_size;
    const uint32_t slabs = logical_tiles / global_chunk_tiles + (logical_tiles % global_chunk_tiles != 0);
    const uint32_t valid_tiles = slabs * region_tiles;
    return valid_tiles < capacity_tiles ? valid_tiles : capacity_tiles;
}

// All sources must participate in the group; configurations with inactive sources
// retain their per-source traversal. This is a scheduling predicate, not a proof
// of mask elision: uniform live slabs can contain globally invalid tail tiles.
constexpr uint32_t packed_kv_source_group_size(
    uint32_t configured, uint32_t ring_size, uint32_t source_tiles, uint32_t logical_tiles, uint32_t active_mask) {
    if (ring_size == 0 || ring_size > 32 || source_tiles == 0 || configured <= 1 || ring_size % configured != 0) {
        return 1;
    }
    const uint32_t all_sources = ~uint32_t{0} >> (32 - ring_size);
    return logical_tiles > 0 && logical_tiles <= source_tiles * ring_size && active_mask == all_sources ? configured
                                                                                                        : 1;
}

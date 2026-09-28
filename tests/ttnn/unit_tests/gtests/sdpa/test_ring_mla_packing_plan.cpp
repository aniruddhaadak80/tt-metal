// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side checks of the split-KV ring MLA contracts shared by the RingJointSDPA program factory,
// reader, compute and writer kernels (ring_mla_packing_plan.hpp, ring_mla_geometry.hpp). Both are
// pure integer math over the block-cyclic KV layout, so they are pinned here without a device.

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_mla_geometry.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_mla_packing_plan.hpp"

namespace {

using ttnn::operations::transformer::sdpa::ring_joint::RingMLAGeometry;

// Global K tile of packed stream tile `stream_tile`: the stream concatenates the live prefix of
// each grouped source, and each source stores its regions of successive global chunks back to back.
uint32_t reference_global_tile(
    uint32_t stream_tile,
    uint32_t source_tiles,
    const std::vector<uint32_t>& source_ids,
    uint32_t region_tiles,
    uint32_t global_chunk_tiles) {
    const uint32_t local = stream_tile % source_tiles;
    return (local / region_tiles) * global_chunk_tiles + source_ids[stream_tile / source_tiles] * region_tiles +
           local % region_tiles;
}

TEST(RingMLAPackingPlan, MaskRunsCoverEveryColumnWithContiguousGlobalRuns) {
    for (uint32_t region : {1u, 2u, 5u}) {
        for (uint32_t ring : {4u, 8u, 32u}) {
            for (uint32_t group : {2u, 4u}) {
                if (ring % group != 0) {
                    continue;
                }
                for (uint32_t slabs : {1u, 3u}) {
                    for (uint32_t chunk_tiles : {1u, 3u, 11u, 20u}) {
                        const uint32_t source_tiles = slabs * region;
                        const uint32_t global_chunk = region * ring;
                        const PackedKVGroupPlan plan{source_tiles, group, chunk_tiles};
                        // A non-trivial source order, as the route delivers it.
                        std::vector<uint32_t> ids;
                        for (uint32_t s = 0; s < group; ++s) {
                            ids.push_back((s * 3 + 1) % ring);
                        }
                        uint32_t covered = 0;
                        for (uint32_t chunk = 0; chunk < plan.chunk_count(); ++chunk) {
                            SCOPED_TRACE(
                                ::testing::Message()
                                << "region=" << region << " ring=" << ring << " group=" << group << " slabs=" << slabs
                                << " chunk_tiles=" << chunk_tiles << " chunk=" << chunk);
                            std::vector<PackedKVMaskRun> runs(chunk_tiles);
                            const uint32_t count = plan.mask_runs(chunk, ids.data(), region, global_chunk, runs.data());
                            ASSERT_GE(count, 1u);
                            ASSERT_LE(count, chunk_tiles);
                            uint32_t begin = 0;
                            for (uint32_t r = 0; r < count; ++r) {
                                ASSERT_GT(runs[r].column_end, begin);
                                for (uint32_t col = begin; col < runs[r].column_end; ++col) {
                                    EXPECT_EQ(
                                        runs[r].global_start_tile + (col - begin),
                                        reference_global_tile(
                                            chunk * chunk_tiles + col, source_tiles, ids, region, global_chunk));
                                }
                                begin = runs[r].column_end;
                            }
                            EXPECT_EQ(begin, plan.valid_tiles(chunk));
                            covered += begin;
                        }
                        EXPECT_EQ(covered, plan.tile_count());
                    }
                }
            }
        }
    }
}

TEST(RingMLAPackingPlan, PartialFinalChunkAndSegments) {
    const PackedKVGroupPlan plan{/*source_tiles=*/5, /*source_count=*/4, /*chunk_tiles=*/8};
    EXPECT_EQ(plan.tile_count(), 20u);
    EXPECT_EQ(plan.chunk_count(), 3u);
    EXPECT_EQ(plan.valid_tiles(0), 8u);
    EXPECT_EQ(plan.valid_tiles(2), 4u);
    EXPECT_EQ(plan.valid_tiles(3), 0u);
    EXPECT_EQ(plan.last_source(0), 1u);  // stream tiles [0, 8) end in source 1
    EXPECT_EQ(plan.last_source(2), 3u);
    // Chunk 1 is stream tiles [8, 16): 2 tiles left in source 1, then 5 in source 2, then 1 in source 3.
    EXPECT_EQ(plan.segment_tiles(1, 0), 2u);
    EXPECT_EQ(plan.segment_tiles(1, 2), 5u);
    EXPECT_EQ(plan.segment_tiles(1, 7), 1u);
}

TEST(RingMLAPackingPlan, ReadinessWaitsOnlyThroughRequiredSourceInOrder) {
    const PackedKVGroupPlan plan{/*source_tiles=*/5, /*source_count=*/4, /*chunk_tiles=*/8};
    PackedKVSourceReadiness readiness;
    std::vector<uint32_t> waited;
    auto wait = [&](uint32_t source) { waited.push_back(source); };
    readiness.wait_for_chunk(plan, 0, wait);
    EXPECT_EQ(waited, (std::vector<uint32_t>{0, 1}));
    readiness.wait_for_chunk(plan, 0, wait);  // reused across Q chunks: no new waits
    EXPECT_EQ(waited.size(), 2u);
    readiness.wait_for_chunk(plan, 2, wait);
    EXPECT_EQ(waited, (std::vector<uint32_t>{0, 1, 2, 3}));
    readiness.drain(4, wait);  // idle cores drain the rest: already complete
    EXPECT_EQ(waited.size(), 4u);
}

TEST(RingMLAPackingPlan, SourceTilesClipToTouchedSlabs) {
    // 8 sources, 2-tile regions: one global chunk is 16 tiles.
    EXPECT_EQ(packed_kv_source_tiles(/*capacity=*/10, /*logical=*/16, 2, 8), 2u);
    EXPECT_EQ(packed_kv_source_tiles(10, 17, 2, 8), 4u);  // a partial second chunk touches slab 2
    EXPECT_EQ(packed_kv_source_tiles(10, 1000, 2, 8), 10u);
    EXPECT_EQ(packed_kv_source_tiles(10, 16, 0, 8), 10u);
}

TEST(RingMLAPackingPlan, GroupOnlyWithEveryConfiguredSourceActive) {
    constexpr uint32_t kAll8 = 0xFFu;
    EXPECT_EQ(packed_kv_source_group_size(4, 8, 2, 16, kAll8), 4u);
    EXPECT_EQ(packed_kv_source_group_size(4, 8, 2, 16, 0x7Fu), 1u);  // one inactive source
    EXPECT_EQ(packed_kv_source_group_size(1, 8, 2, 16, kAll8), 1u);  // not configured
    EXPECT_EQ(packed_kv_source_group_size(3, 8, 2, 16, kAll8), 1u);  // does not divide the ring
    EXPECT_EQ(packed_kv_source_group_size(4, 8, 2, 17, kAll8), 1u);  // logical beyond capacity
    EXPECT_EQ(packed_kv_source_group_size(4, 8, 2, 0, kAll8), 1u);
    EXPECT_EQ(packed_kv_source_group_size(4, 32, 2, 64, 0xFFFFFFFFu), 4u);
}

TEST(RingMLAGeometry, ValidRequiresWholeRegionsPerSource) {
    // Galaxy SP8 x TP4: 20-tile Q slab, 5-tile regions, 11 slabs of cache per source.
    EXPECT_TRUE((RingMLAGeometry{8, 32, 20, 55}.valid()));
    EXPECT_TRUE((RingMLAGeometry{8, 8, 20, 20}.valid()));    // classic: one region per Q slab
    EXPECT_FALSE((RingMLAGeometry{8, 32, 20, 54}.valid()));  // capacity is not whole regions
    EXPECT_FALSE((RingMLAGeometry{8, 32, 18, 54}.valid()));  // Q slab does not split into 4 stripes
    EXPECT_FALSE((RingMLAGeometry{8, 12, 20, 20}.valid()));  // KV sources not a multiple of Q shards
    EXPECT_FALSE((RingMLAGeometry{8, 64, 16, 16}.valid()));  // beyond the 32-source mask
    EXPECT_FALSE((RingMLAGeometry{0, 32, 20, 55}.valid()));
    EXPECT_FALSE((RingMLAGeometry{8, 32, 0, 55}.valid()));
    EXPECT_FALSE((RingMLAGeometry{8, 32, 20, 0}.valid()));
}

}  // namespace

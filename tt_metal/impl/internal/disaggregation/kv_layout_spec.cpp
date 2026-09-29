// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/kv_layout_spec.hpp>

namespace tt::tt_metal::internal::disaggregation {

uint32_t chunk_size_bytes(const MemLayout& mem, uint32_t tokens_per_chunk, uint64_t feature_dim) {
    if (mem.bf16_chunk_bytes) {
        return static_cast<uint32_t>(tokens_per_chunk * feature_dim * kBf16Bytes);
    }
    return static_cast<uint32_t>((tokens_per_chunk / kTile) * (feature_dim / kTile) * kBfp8TileBytes);
}

InnerFootprint KvLayoutSpec::inner_footprint() const {
    const auto& shape = tensor.logical_shape();
    const uint32_t rank = static_cast<uint32_t>(shape.rank());
    const std::optional<uint32_t> seq = seq_axis.has_value() ? std::optional<uint32_t>(seq_axis->get()) : std::nullopt;
    const std::optional<uint32_t> head =
        head_shard_axis.has_value() ? std::optional<uint32_t>(head_shard_axis->get()) : std::nullopt;

    // Exclude the sequence axis, the head-shard axis, and (for rank>2 tensors whose seq axis is not
    // axis 0) the leading batch/slot axis — leaving the per-token feature width.
    const bool exclude_batch = rank > 2 && seq != std::optional<uint32_t>(0);

    uint64_t f = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        if (seq.has_value() && i == *seq) {
            continue;
        }
        if (head.has_value() && i == *head) {
            continue;
        }
        if (exclude_batch && i == 0) {
            continue;
        }
        f *= static_cast<uint64_t>(shape[i]);
    }
    return InnerFootprint{f};
}

}  // namespace tt::tt_metal::internal::disaggregation

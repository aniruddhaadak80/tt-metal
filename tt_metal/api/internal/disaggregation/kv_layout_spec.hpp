// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <variant>
#include <vector>

#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::internal::disaggregation {

// KvLayoutSpec — the model-facing, declarative definition of ONE cached KV/state tensor.
//
// It is the "common language" a model uses to specify its KV cache layout (see README consumer
// #1); the migration, tiering, and weights services consume it. It answers *what a tensor is* and
// *how it is addressed* — deliberately residence-agnostic. Deriving physical addresses from it (the
// old per-model `locate`), the sender/receiver resharding planner, and tiering are downstream KV
// Manager concerns, not part of this definition.
//
// A spec composes four independent parts over one native `TensorSpec`:
//   (1) TemporalPolicy — how the universal `prefix_len` coordinate is interpreted for this tensor
//   (2) Distribution   — per-tensor-axis placement onto the device mesh
//   (3) MemLayout      — intra-device physical layout (DRAM bank ordering)
//   (4) AddressingMode — slot-direct vs. paged (block-table) indirection
//
// Each part varies independently: a new attention/mixer type is a new TemporalPolicy, a new
// parallelization is a new Distribution — neither touches the others.
//
// SKELETON: the parts and their fields are established here; behaviour (e.g. inner_footprint()
// derivation, validation, the named-parallelism lowering) is filled in incrementally.

// ---------------------------------------------------------------------------------------------
// Strong types — semantic values that must never be interchanged (see cpp coding standards).
// ---------------------------------------------------------------------------------------------

// Index of the sequence (temporal) axis within the tensor's logical shape.
using SeqAxis = ttsl::StrongType<uint32_t, struct SeqAxisTag>;
// Index of an axis of the device mesh (a Shard target).
using MeshAxis = ttsl::StrongType<uint32_t, struct MeshAxisTag>;
// Sliding-window width W, in tokens.
using WindowTokens = ttsl::StrongType<uint32_t, struct WindowTokensTag>;
// Block-/chunk-local attention span, in tokens.
using ChunkTokens = ttsl::StrongType<uint32_t, struct ChunkTokensTag>;
// Depth of a conv / token-shift rolling ring: kernel_size - 1 inputs retained.
using RollingWidth = ttsl::StrongType<uint32_t, struct RollingWidthTag>;
// F — the per-token inner footprint: the FEATURE width, product of the non-sequence axis extents
// EXCLUDING the leading batch/slot axis and the head-shard axis (both are placement/paging concerns,
// not part of the per-token footprint). When there is no sequence axis (recurrent / conv summaries)
// this is the whole per-slot state size.
using InnerFootprint = ttsl::StrongType<uint64_t, struct InnerFootprintTag>;

// Index of the tensor axis carrying the attention head (the head-shard axis), when a scheme fans a
// tensor out over heads across the mesh (BLOCK/CYCLIC GQA). Excluded from F.
using HeadShardAxis = ttsl::StrongType<uint32_t, struct HeadShardAxisTag>;
// CP (context-parallel) stride: tokens each sequence-shard device owns per round before the seq axis
// rotates to the next device. Lives in the flash op / migration path, not the allocated TensorSpec.
using DeviceChunkSize = ttsl::StrongType<uint32_t, struct DeviceChunkSizeTag>;
// Origin offset of the first sequence-shard (CP) device on the mesh axis.
using SpOrigin = ttsl::StrongType<uint32_t, struct SpOriginTag>;
// K-chunk (DRAM page / block) size, in tokens — the flash op's effective SDPA k-chunk / block_size.
using KChunkSize = ttsl::StrongType<uint32_t, struct KChunkSizeTag>;
// Number of DRAM banks a single head/group fans out over (BLOCK / CYCLIC height-sharding).
using BanksPerHead = ttsl::StrongType<uint32_t, struct BanksPerHeadTag>;
// index_k column-split degree (BLOCK_CYCLIC): how many devices a sparse-index row is split across.
using IdxCp = ttsl::StrongType<uint32_t, struct IdxCpTag>;
// GQA group index -> mesh ROW block (BLOCK K/V per-group placement).
using GqaGroup = ttsl::StrongType<uint32_t, struct GqaGroupTag>;
// Extent of the sequence-shard (CP) mesh axis.
using SpDim = ttsl::StrongType<uint32_t, struct SpDimTag>;
// Mesh column count (TP fan-out).
using MeshCols = ttsl::StrongType<uint32_t, struct MeshColsTag>;
// Mesh row count.
using MeshRows = ttsl::StrongType<uint32_t, struct MeshRowsTag>;
// A layer index within the model.
using LayerIndex = ttsl::StrongType<uint32_t, struct LayerIndexTag>;

// ---------------------------------------------------------------------------------------------
// (1) TemporalPolicy — retention window over the prefix, keyed by the universal `prefix_len`.
// Each alternative fixes (extent, address-map, update) for the tensor's sequence dimension.
// ---------------------------------------------------------------------------------------------
namespace temporal {

// [0, prefix_len) — grows with the prefix. Full KV, MLA latent.
struct Dense {};

// [prefix_len - width, prefix_len) — a rolling ring; addressed by (position mod width). SWA.
struct Window {
    WindowTokens width;
};

// The current chunk only; resets at chunk boundaries (not a trailing window). Llama-4 iRoPE local.
struct BlockLocal {
    ChunkTokens chunk;
};

// The last (width) inputs — a tiny conv / token-shift ring. Mamba conv1d, RWKV token-shift.
struct Rolling {
    RollingWidth width;
};

// Computed once at prefill and frozen; addressed by SOURCE position, never grows. Enc-dec /
// vision cross-attention.
struct Static {};

// No sequence axis at all — a fixed-size recurrent summary addressed by a sparse checkpoint tag
// (the prefix_len it was snapshotted at). Mamba SSM state, KDA state matrix.
struct None {};

}  // namespace temporal

using TemporalPolicy = std::variant<
    temporal::Dense,
    temporal::Window,
    temporal::BlockLocal,
    temporal::Rolling,
    temporal::Static,
    temporal::None>;

// ---------------------------------------------------------------------------------------------
// (2) Distribution — placement of each logical tensor axis onto the device mesh. Named
// parallelisms (TP/PP/DP/SP/EP) lower to this: "shard tensor dim X onto mesh axis Y, or replicate".
// Only Shard/Replicate matter for a stored tensor; collectives are a compute concern.
// ---------------------------------------------------------------------------------------------

// This tensor axis is sharded across the given mesh axis.
struct Shard {
    MeshAxis mesh_axis;
};

// This tensor axis is replicated (same bytes on every device along its mesh axes). MLA latent /
// MQA use this on the head axis.
struct Replicate {};

using AxisPlacement = std::variant<Shard, Replicate>;

// One placement per logical tensor axis, index-aligned with the TensorSpec's logical shape.
struct Distribution {
    std::vector<AxisPlacement> per_axis;
};

// ---------------------------------------------------------------------------------------------
// (3) MemLayout — intra-device physical layout. The TensorSpec's MemoryConfig already implies the
// bank striping (interleaved / NdShardSpec); this selects the bank ordering permutation.
// ---------------------------------------------------------------------------------------------

enum class BankOrder : uint8_t {
    Identity = 0,  // portable round-robin (page_id % num_banks)
    Optimal = 1,   // NOC-local permutation co-locating each bank with its consuming cores
};

// How a chunk index maps to a (bank, per-bank offset). GENERATION POLICY the allocated TensorSpec
// does not carry — it comes from the flash op / migration path (see kv_layout_spec_smoke/README).
//   Natural      page grows monotonically; bank = page_id % num_banks; per-slot pages stack.
//   MlaShard     shard_id = slot*chunks_per_slot + local_chunk; bank = perm[shard_id % banks];
//                per-bank offset stacks by shard_id / banks. (MLA latent cache)
//   Block        height-sharded per head/group; bank_order[base + tile_row / st_pb].
//   Cyclic       bank_order[base + (tile_row / sk_chunk_t) % bph]; within stacks by chunk.
//   BlockCyclic  round-robin blocks over banks; bank_order[global_block % num_banks]. (index_k)
enum class BankScheme : uint8_t {
    Natural = 0,
    MlaShard = 1,
    Block = 2,
    Cyclic = 3,
    BlockCyclic = 4,
};

// The OPTIMAL DRAM bank permutation (NOC-local ordering); index i -> physical bank id.
inline constexpr std::array<uint32_t, 8> kOptimalDramBankOrder = {1, 3, 2, 0, 5, 7, 6, 4};
inline constexpr uint32_t kNumDramBanks = 8;
inline constexpr uint32_t kTile = 32;
inline constexpr uint32_t kBfp8TileBytes = 1088;  // 32x32 bfloat8_b tile
inline constexpr uint32_t kBf16Bytes = 2;

struct MemLayout {
    BankOrder bank_order = BankOrder::Optimal;
    BankScheme bank_scheme = BankScheme::Natural;

    // Generation-policy fields (not derivable from the allocated tensor).
    KChunkSize k_chunk_size{128};    // tokens per DRAM page / block
    ChunkTokens chunk_n_tokens{kTile};  // migration granule (tokens per chunk)
    uint32_t num_banks = kNumDramBanks;
    uint32_t num_blocks = kNumDramBanks;         // permutation block count (OPTIMAL indexer)
    BanksPerHead banks_per_head{kNumDramBanks};  // BLOCK/CYCLIC per-head fan-out

    // Element sizing: bfp8 (tiled) chunks vs bf16 (dense) chunks.
    bool bf16_chunk_bytes = false;  // false => bfp8 tiled sizing
};

// chunk_size_bytes for one (tokens_per_chunk x feature_dim) chunk under the layout's dtype sizing.
uint32_t chunk_size_bytes(const MemLayout& mem, uint32_t tokens_per_chunk, uint64_t feature_dim);

// ---------------------------------------------------------------------------------------------
// (4) AddressingMode — how a logical position resolves to a physical slot.
// ---------------------------------------------------------------------------------------------

enum class AddressingMode : uint8_t {
    Slot = 0,   // direct: physical = f(slot, coord), no runtime indirection
    Paged = 1,  // a block table maps logical position -> physical block (enables sharing / packing)
};

// ---------------------------------------------------------------------------------------------
// The composed spec.
// ---------------------------------------------------------------------------------------------

struct KvLayoutSpec {
    // The logical tensor: shape + dtype + layout + memory config. The sequence axis is OPTIONAL —
    // present for attention (Dense/Window/BlockLocal/Static), absent for recurrent/conv summaries
    // (Rolling/None), which have no per-token dimension.
    TensorSpec tensor;
    std::optional<SeqAxis> seq_axis;

    TemporalPolicy temporal;
    Distribution distribution;
    MemLayout mem_layout;
    AddressingMode addressing = AddressingMode::Slot;

    // Mesh-axis extents + CP context (generation policy; not carried by the allocated TensorSpec).
    SpDim sp_dim{1};        // extent of the seq-shard (CP) mesh axis
    MeshCols mesh_cols{1};
    MeshRows mesh_rows{1};
    SpOrigin sp_origin{0};
    // CP stride on Shard(seq). When unset, defaults to k_chunk_size * num_banks.
    std::optional<DeviceChunkSize> device_chunk_size;
    IdxCp idx_cp{1};  // index_k column-split degree (BLOCK_CYCLIC)

    // Placement helpers.
    std::optional<GqaGroup> group;               // GQA group -> mesh ROW block (BLOCK K/V)
    std::optional<HeadShardAxis> head_shard_axis;  // tensor axis carrying head (BLOCK/CYCLIC per-head)

    // Per-layer overrides + applicability. `layers` empty => resident on every enumerated layer.
    std::set<uint32_t> layers;
    std::map<uint32_t, TemporalPolicy> temporal_by_layer;
    std::map<uint32_t, BankScheme> bank_scheme_by_layer;
    std::map<uint32_t, KChunkSize> k_chunk_by_layer;

    bool applies_to_layer(uint32_t layer) const { return layers.empty() || layers.count(layer) != 0; }

    // Whether this tensor has a per-token sequence axis (attention family) vs. is a fixed-size
    // summary (recurrent / conv).
    bool has_sequence() const { return seq_axis.has_value(); }

    // F — the per-token FEATURE width: product of the non-sequence axis extents EXCLUDING the leading
    // batch/slot axis and the head-shard axis. Derived from `tensor`, `seq_axis`, `head_shard_axis`.
    InnerFootprint inner_footprint() const;
};

}  // namespace tt::tt_metal::internal::disaggregation

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "mesh_device,device_params,topology",
    [
        ((2, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 2**20}, ttnn.Topology.Linear),
        ((1, 8), {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 2**20}, ttnn.Topology.Ring),
    ],
    indirect=["mesh_device", "device_params"],
    ids=["linear_2x4", "ring_1x8"],
)
@pytest.mark.parametrize(
    "op,dim",
    [("all_gather", 3), ("all_gather_sharded", 3), ("reduce_scatter", 0), ("reduce_scatter", 3), ("all_reduce", 3)],
)
@pytest.mark.parametrize("use_barrier", [False, True])
def test_ccl_dispatch_cache_rebinding(mesh_device, op, dim, topology, use_barrier):
    """Rebind inputs/semaphores/outputs before and after trace payload relocation."""
    group_size = mesh_device.shape[1]
    num_devices = mesh_device.get_num_devices()
    if op == "all_gather_sharded" and group_size != 4:
        pytest.skip("Llama sharded specialization requires a four-device gather")
    torch.manual_seed(0)
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    is_gather = op.startswith("all_gather")
    input_memory = output_memory = ttnn.DRAM_MEMORY_CONFIG
    local_shape = (group_size, 1, 32, group_size * 32)
    if op == "all_gather_sharded":
        local_shape = (1, 1, 32, 32)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        input_memory, output_memory = [
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(shard_grid, [32, width], ttnn.ShardOrientation.ROW_MAJOR),
            )
            for width in (32, 128)
        ]
    intermediate_memory = None
    if op == "all_reduce":
        local_shape = (1, 1, 32, 128)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        input_memory = output_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(shard_grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        intermediate_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(shard_grid, [32, 32 * group_size], ttnn.ShardOrientation.ROW_MAJOR),
        )
    intermediates = []
    inputs, goldens, semaphores, barriers = [], [], [], []
    for iteration in range(2):
        shards = [
            (torch.randint(0, 4, local_shape) + chip + iteration * 8).to(torch.bfloat16) for chip in range(num_devices)
        ]
        inputs.append(
            ttnn.from_torch(
                torch.cat(shards, dim=dim),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=input_memory,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
            )
        )
        expected = []
        for row in range(mesh_device.shape[0]):
            group = shards[row * group_size : (row + 1) * group_size]
            if is_gather:
                expected.extend([torch.cat(group, dim=dim)] * group_size)
            elif op == "all_reduce":
                expected.extend([torch.stack(group).float().sum(0).to(torch.bfloat16)] * group_size)
            else:
                expected.extend(torch.stack(group).float().sum(0).to(torch.bfloat16).chunk(group_size, dim=dim))
        if op == "all_reduce":
            intermediates.append(
                ttnn.empty(
                    [1, 1, 32, 128 * group_size],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=intermediate_memory,
                )
            )
        goldens.append(expected)
        semaphores.append([ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(3)])
        barriers.append(ttnn.create_global_semaphore(mesh_device, cores, 0) if use_barrier else None)

    # These cases must run their named topology; a fallback is not Ring coverage.
    assert ttnn.get_usable_topology(inputs[0], topology=topology, cluster_axis=1) == topology

    def run(index):
        if op == "all_reduce":
            return ttnn.experimental.all_reduce_async(
                inputs[index],
                intermediates[index],
                cluster_axis=1,
                mesh_device=mesh_device,
                multi_device_global_semaphore=semaphores[index][0],
                memory_config=output_memory,
                topology=topology,
                num_links=1,
            )
        kwargs = dict(
            dim=dim,
            cluster_axis=1,
            num_links=1,
            topology=topology,
            memory_config=output_memory,
            multi_device_global_semaphore=semaphores[index][:2] if is_gather else semaphores[index],
            barrier_semaphore=barriers[index],
            num_workers_per_link=2,
        )
        if is_gather:
            return ttnn.experimental.all_gather_async(inputs[index], **kwargs)
        return ttnn.experimental.reduce_scatter_minimal_async(inputs[index], **kwargs)

    def check(output, index):
        ttnn.synchronize_device(mesh_device)
        for actual, expected in zip(ttnn.get_device_tensors(output), goldens[index], strict=True):
            torch.testing.assert_close(ttnn.to_torch(actual), expected, rtol=0, atol=0)

    mesh_device.enable_program_cache()
    # Keep both outputs live so that a cache hit must update the output binding too.
    outputs = [run(0)]
    check(outputs[0], 0)
    entries = mesh_device.num_program_cache_entries()
    outputs.append(run(1))
    check(outputs[1], 1)
    assert mesh_device.num_program_cache_entries() == entries

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = run(1)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            check(traced_output, 1)
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    check(run(0), 0)
    check(run(1), 1)
    assert mesh_device.num_program_cache_entries() == entries


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
def test_ccl_topology_cache_reshape(mesh_device):
    def query(device, axis):
        tensor = ttnn.empty([1, 1, 32, 32], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return ttnn.get_usable_topology(tensor, topology=ttnn.Topology.Ring, cluster_axis=axis)

    for shape in [(2, 4), (1, 8), (2, 4)]:
        mesh_device.reshape(ttnn.MeshShape(shape))
        # Query the reshaped mesh before a fresh mesh ID replaces the thread-local cache entry.
        actual = query(mesh_device, 1)
        reference = mesh_device.create_submesh(ttnn.MeshShape(shape))
        try:
            assert actual == query(reference, 1)
        finally:
            ttnn.close_mesh_device(reference)
        # Leave the parent cached before the next reshape; a two-device axis must stay linear.
        assert query(mesh_device, 1) == actual
        if shape[0] == 2:
            assert query(mesh_device, 0) == ttnn.Topology.Linear

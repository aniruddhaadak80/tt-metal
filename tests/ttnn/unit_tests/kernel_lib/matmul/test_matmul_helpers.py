# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exercise compute helpers directly, without production matmul kernels or factories."""

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device
KERNEL = "tests/ttnn/unit_tests/kernel_lib/matmul/kernels/matmul.cpp"


def _tiles(matrix):
    rows, cols = matrix.shape
    return matrix.reshape(rows // 32, 32, cols // 32, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)


@pytest.mark.parametrize("k_blocks", [1, 3], ids=["single-k", "spill-reload"])
@pytest.mark.parametrize("l1_acc", [False, True], ids=["software", "l1-acc"])
@pytest.mark.parametrize(
    "post_op",
    [0, 1, 2, 3, 4, 7],
    ids=[
        "plain",
        "bias",
        "relu",
        "relu6",
        "bias-relu6",
        "bias-relu",
    ],
)
@pytest.mark.parametrize("static_shape", [False, True], ids=["runtime-shape", "static-shape"])
def test_matmul_helpers(device, k_blocks, l1_acc, post_op, static_shape, batches=1, same_cb=False):
    # Signed, exactly representable inputs distinguish tile order and ensure ReLU
    # must happen after the K reduction. Different K slices also catch lost partials.
    generator = torch.Generator().manual_seed(42)
    a = torch.randint(-2, 3, (128 * batches, 32 * k_blocks), generator=generator).float()
    b = torch.zeros(32 * k_blocks, 128)
    for k in range(k_blocks):
        for n in range(4):
            b[k * 32 : (k + 1) * 32, n * 32 : (n + 1) * 32] = torch.eye(32) * (1 if (k + n) % 2 else -1)
    expected = a @ b
    bias = torch.zeros(32, 128)
    bias[0] = torch.arange(128) % 5 - 2
    if post_op in (1, 4, 7):
        expected += bias[0]
    if post_op in (2, 7):
        expected = expected.relu()
    elif post_op in (3, 4):
        expected = expected.clamp(0, 6)

    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    def memory(shape):
        return ttnn.create_sharded_memory_config(
            shape,
            core_grid=cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def tensor(tiles):
        physical = tiles.reshape(-1, 32)
        return ttnn.from_torch(
            physical,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory(tuple(physical.shape)),
        )

    # The helper consumes M x block_K and block_K x N tiles for each K block.
    ta = tensor(
        torch.cat(
            [
                _tiles(a[batch * 128 : (batch + 1) * 128, k * 32 : (k + 1) * 32])
                for batch in range(batches)
                for k in range(k_blocks)
            ]
        )
    )
    tb = tensor(_tiles(b).repeat(batches, 1, 1))
    t_bias = tensor(_tiles(bias))
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape((16 * batches * 32, 32)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory((16 * batches * 32, 32))
    )
    page = ttnn.tile_size(ttnn.bfloat16)
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(i, t) for i, t in [(0, ta), (1, tb), (3, t_bias), (16, out)]]
    cbs.append(
        ttnn.CBDescriptor(
            total_size=16 * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=2, data_format=ttnn.bfloat16, page_size=page)],
        )
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[
            k_blocks,
            int(l1_acc),
            post_op,
            int(static_shape),
            batches,
            int(same_cb),
        ],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
    )
    result = ttnn.generic_op([ta, tb, t_bias, out], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs))
    actual_tiles = ttnn.to_torch(result).reshape(16 * batches, 32, 32).float()
    expected_tiles = _tiles(expected)
    order = [
        batch * 16 + r * 4 + c
        for batch in range(batches)
        for br in (0, 2)
        for bc in (0, 2)
        for r in range(br, br + 2)
        for c in range(bc, bc + 2)
    ]
    expected_tiles = expected_tiles[order]
    torch.testing.assert_close(actual_tiles, expected_tiles, rtol=0, atol=0)


@pytest.mark.parametrize("l1_acc", [False, True])
def test_matmul_activation_batches(device, l1_acc):
    test_matmul_helpers(device, 3, l1_acc, 3, False, batches=2)


@pytest.mark.parametrize("k_blocks", [1, 3])
@pytest.mark.parametrize("l1_acc", [False, True])
@pytest.mark.parametrize("post_op", [0, 2, 3], ids=["plain", "relu", "relu6"])
def test_matmul_same_output_and_partials(device, k_blocks, l1_acc, post_op):
    test_matmul_helpers(device, k_blocks, l1_acc, post_op, False, same_cb=True)

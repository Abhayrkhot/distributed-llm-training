import numpy as np
import pytest
import torch

from dlt.data.packed import GlobalOrder, PackedTokens


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "train.bin"
    np.arange(1000, dtype=np.uint32).tofile(path)
    return path


def test_packed_samples_are_disjoint_shifted_windows(token_file):
    ds = PackedTokens(token_file, seq_len=10)
    assert len(ds) == 99  # (1000 - 1) // 10
    x, y = ds.batch(torch.tensor([0, 3]))
    assert x.shape == y.shape == (2, 10)
    assert x[0].tolist() == list(range(10)) and y[0].tolist() == list(range(1, 11))
    assert x[1, 0].item() == 30 and y[1, -1].item() == 40


def test_every_sample_once_per_epoch_across_ranks_and_micro_batches():
    order = GlobalOrder(n_samples=96, global_batch=16, seed=7)
    world, micro = 2, 4
    seen = []
    for step in range(order.steps_per_epoch):
        for rank in range(world):
            for mb in order.micro_batches(step, rank, world, micro):
                assert len(mb) == micro
                seen.extend(mb.tolist())
    assert sorted(seen) == list(range(96))


def test_order_is_a_pure_function_of_seed_and_step():
    a = GlobalOrder(1000, 32, seed=1)
    b = GlobalOrder(1000, 32, seed=1)
    assert torch.equal(a.global_indices(40), b.global_indices(40))  # 40 is in epoch 1
    assert not torch.equal(a.global_indices(0), GlobalOrder(1000, 32, seed=2).global_indices(0))
    assert not torch.equal(a.global_indices(0), a.global_indices(a.steps_per_epoch))  # new epoch, new order


def test_single_rank_and_two_ranks_cover_the_same_global_batch():
    order = GlobalOrder(512, 32, seed=3)
    one = torch.cat(order.micro_batches(5, 0, 1, 8))
    two = torch.cat([torch.cat(order.micro_batches(5, r, 2, 8)) for r in range(2)])
    assert torch.equal(one, two)


def test_rejects_indivisible_batches():
    order = GlobalOrder(512, 30, seed=0)
    with pytest.raises(ValueError):
        order.micro_batches(0, 0, 4, 2)

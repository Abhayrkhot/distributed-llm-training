"""Every shipped training config must run on 1 and 2 GPUs (the Explorer limit) without edits."""

from pathlib import Path

import pytest

from dlt.train.config import PINNED, parse

CONFIGS = sorted((Path(__file__).resolve().parent.parent / "configs" / "training").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_runs_on_one_and_two_gpus(path):
    cfg = parse(["--config", str(path), "--set", "data=/dev/null"])
    assert cfg.init != "pretrained" or cfg.model in PINNED, "pretrained models must have a pinned revision"
    for world in (1, 2):
        assert cfg.global_batch % (world * cfg.micro_batch) == 0, (
            f"global_batch does not split over {world} GPUs"
        )

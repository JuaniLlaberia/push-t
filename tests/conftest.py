import os

import pytest
import torch

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

@pytest.fixture(scope="session")
def device() -> torch.device:
    from pusht_bc.utils import get_device

    return get_device()

@pytest.fixture(scope="session")
def pusht_dataset():
    from pusht_bc.data import load_dataset

    return load_dataset("lerobot/pusht")

@pytest.fixture()
def env():
    from pusht_bc.evaluate import make_env

    e = make_env(max_steps=300)
    yield e
    e.close()

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    c["model"].update(epochs=3, hidden_size=32, max_train_samples_per_epoch=3000)
    return c


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory):
    from src.data.synthetic import make_demo_dataset
    d = tmp_path_factory.mktemp("raw")
    make_demo_dataset(d, n_days=3, n_windows=700, seed=11)
    return d


@pytest.fixture(scope="session")
def states(demo_dir, cfg):
    from src.data.windows import build_states
    st, _ = build_states(sorted(demo_dir.glob("*.csv")), cfg, progress=None)
    return st


@pytest.fixture(scope="session")
def trained(states, cfg, tmp_path_factory):
    from src.models.train import train_from_states
    out = tmp_path_factory.mktemp("models")
    r = train_from_states(states, cfg, out, synthetic=True, log=lambda *_: None)
    return out, r

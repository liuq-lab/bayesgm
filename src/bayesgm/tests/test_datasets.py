import numpy as np
import pandas as pd
import pytest
from bayesgm.datasets.base_sampler import Base_sampler
from bayesgm.datasets.causal_samplers import Semi_acic_sampler, Sim_Hirano_Imbens_sampler, Sim_Sun_sampler, Sim_Colangelo_sampler

@pytest.fixture
def base_sampler_data():
    """Fixture for creating dummy data for Base_sampler."""
    x = np.random.rand(100, 1)
    y = np.random.rand(100, 1)
    v = np.random.rand(100, 5)
    return x, y, v

def test_base_sampler_init(base_sampler_data):
    """Test initialization of Base_sampler."""
    x, y, v = base_sampler_data
    sampler = Base_sampler(x, y, v, batch_size=10, normalize=False)
    assert sampler.data_x.shape == (100, 1)
    assert sampler.data_y.shape == (100, 1)
    assert sampler.data_v.shape == (100, 5)
    assert sampler.batch_size == 10

def test_base_sampler_next_batch(base_sampler_data):
    """Test next_batch method of Base_sampler."""
    x, y, v = base_sampler_data
    sampler = Base_sampler(x, y, v, batch_size=10, normalize=False)
    batch_x, batch_y, batch_v = sampler.next_batch()
    assert batch_x.shape == (10, 1)
    assert batch_y.shape == (10, 1)
    assert batch_v.shape == (10, 5)

def test_base_sampler_load_all(base_sampler_data):
    """Test load_all method of Base_sampler."""
    x, y, v = base_sampler_data
    sampler = Base_sampler(x, y, v, batch_size=10, normalize=False)
    loaded_x, loaded_y, loaded_v = sampler.load_all()
    assert np.allclose(x, loaded_x)
    assert np.allclose(y, loaded_y)
    assert np.allclose(v, loaded_v)

def test_semi_acic_sampler(tmp_path):
    """Test Semi_acic_sampler."""
    # Create dummy ACIC dataset
    path = tmp_path / "acic_data"
    path.mkdir()
    covariants_file = path / "x.csv"
    factuals_dir = path / "scaling" / "factuals"
    factuals_dir.mkdir(parents=True)
    factuals_file = factuals_dir / "d5bd8e4814904c58a79d7cdcd7c2a1bb.csv"

    # Create dummy data
    df_covariants = pd.DataFrame(np.random.rand(100, 5), columns=["v0", "v1", "v2", "v3", "v4"])
    df_covariants.index.name = "sample_id"
    df_covariants.to_csv(covariants_file)

    df_sim = pd.DataFrame({
        "z": np.random.rand(100),
        "y": np.random.rand(100),
    })
    df_sim.index.name = "sample_id"
    df_sim.to_csv(factuals_file)

    # Test Semi_acic_sampler
    sampler = Semi_acic_sampler(batch_size=10, path=str(path), ufid="d5bd8e4814904c58a79d7cdcd7c2a1bb")
    batch_x, batch_y, batch_v = sampler.next_batch()
    assert batch_x.shape == (10, 1)
    assert batch_y.shape == (10, 1)
    assert batch_v.shape == (10, 5)

def test_sim_hirano_imbens_sampler():
    """Test Sim_Hirano_Imbens_sampler."""
    sampler = Sim_Hirano_Imbens_sampler(batch_size=10, N=100, v_dim=5, seed=0)
    batch_x, batch_y, batch_v = sampler.next_batch()
    assert batch_x.shape == (10, 1)
    assert batch_y.shape == (10, 1)
    assert batch_v.shape == (10, 5)

    # Test load_all method
    loaded_x, loaded_y, loaded_v = sampler.load_all()
    assert loaded_x.shape == (100, 1)
    assert loaded_y.shape == (100, 1)
    assert loaded_v.shape == (100, 5)

def test_sim_sun_sampler():
    """Test Sim_Sun_sampler."""
    sampler = Sim_Sun_sampler(batch_size=10, N=100, v_dim=10, seed=0)
    batch_x, batch_y, batch_v = sampler.next_batch()
    assert batch_x.shape == (10, 1)
    assert batch_y.shape == (10, 1)
    assert batch_v.shape == (10, 10)

    # Test load_all method
    loaded_x, loaded_y, loaded_v = sampler.load_all()
    assert loaded_x.shape == (100, 1)
    assert loaded_y.shape == (100, 1)
    assert loaded_v.shape == (100, 10)

def test_sim_colangelo_sampler():
    """Test Sim_Colangelo_sampler."""
    sampler = Sim_Colangelo_sampler(batch_size=10, N=100, v_dim=5, seed=0)
    batch_x, batch_y, batch_v = sampler.next_batch()
    assert batch_x.shape == (10, 1)
    assert batch_y.shape == (10, 1)
    assert batch_v.shape == (10, 5)

    # Test load_all method
    loaded_x, loaded_y, loaded_v = sampler.load_all()
    assert loaded_x.shape == (100, 1)
    assert loaded_y.shape == (100, 1)
    assert loaded_v.shape == (100, 5)
    
import numpy as np
import pandas as pd
import os
import pytest
from bayesgm.utils.data_io import save_data, parse_file, parse_file_triplet

def test_save_data_npy(tmp_path):
    """Test saving data as a .npy file."""
    fname = tmp_path / "test_data.npy"
    data = np.random.rand(10, 5)
    save_data(str(fname), data)
    # Assert file exists
    assert fname.exists()
    # Assert data integrity
    loaded_data = np.load(fname)
    assert np.allclose(data, loaded_data)

def test_save_data_txt(tmp_path):
    """Test saving data as a .txt file."""
    fname = tmp_path / "test_data.txt"
    data = np.random.rand(10, 5)
    save_data(str(fname), data)
    # Assert file exists
    assert fname.exists()
    # Assert data integrity
    loaded_data = np.loadtxt(fname)
    assert np.allclose(data, loaded_data, atol=1e-6)

def test_save_data_csv(tmp_path):
    """Test saving data as a .csv file."""
    fname = tmp_path / "test_data.csv"
    data = np.random.rand(10, 5)
    save_data(str(fname), data, delimiter='\t')
    # Assert file exists
    assert fname.exists()
    # Assert data integrity
    loaded_data = np.loadtxt(fname, delimiter='\t')
    assert np.allclose(data, loaded_data, atol=1e-6)

def test_save_data_invalid_format(tmp_path):
    """Test saving data with an invalid file extension."""
    fname = tmp_path / "test_data.invalid"
    data = np.random.rand(10, 5)
    with pytest.raises(ValueError, match="Wrong saving format, please specify either .npy, .txt, or .csv"):
        save_data(str(fname), data)

def test_parse_file_triplet_npz(tmp_path):
    """Test parsing data from an .npz file with parse_file_triplet."""
    fname = tmp_path / "test_data.npz"
    data_x = np.random.rand(10, 1).astype('float32')
    data_y = np.random.rand(10, 1).astype('float32')
    data_v = np.random.rand(10, 3).astype('float32')
    np.savez(fname, x=data_x, y=data_y, v=data_v)

    parsed_x, parsed_y, parsed_v = parse_file_triplet(str(fname), normalize=False)
    assert np.allclose(data_x, parsed_x)
    assert np.allclose(data_y, parsed_y)
    assert np.allclose(data_v, parsed_v)

def test_parse_file_triplet_csv(tmp_path):
    """Test parsing data from a .csv file with parse_file_triplet."""
    fname = tmp_path / "test_data.csv"
    data = np.hstack([np.random.rand(10, 1), np.random.rand(10, 1), np.random.rand(10, 3)])
    columns = ["X", "Y"] + [f"V{i}" for i in range(data.shape[1] - 2)]
    pd.DataFrame(data, columns=columns).to_csv(fname, index=False, sep='\t')

    parsed_x, parsed_y, parsed_v = parse_file_triplet(str(fname), sep='\t', header=0, normalize=False)
    assert np.allclose(data[:, 0].reshape(-1, 1), parsed_x)
    assert np.allclose(data[:, 1].reshape(-1, 1), parsed_y)
    assert np.allclose(data[:, 2:], parsed_v)

def test_parse_file_triplet_txt(tmp_path):
    """Test parsing data from a .txt file with parse_file_triplet."""
    fname = tmp_path / "test_data.txt"
    data = np.hstack([np.random.rand(10, 1), np.random.rand(10, 1), np.random.rand(10, 3)])
    np.savetxt(fname, data, delimiter='\t')

    parsed_x, parsed_y, parsed_v = parse_file_triplet(str(fname), sep='\t', normalize=False)
    assert np.allclose(data[:, 0].reshape(-1, 1), parsed_x)
    assert np.allclose(data[:, 1].reshape(-1, 1), parsed_y)
    assert np.allclose(data[:, 2:], parsed_v)

def test_parse_file_triplet_invalid_format(tmp_path):
    """Test parsing data from an unsupported file format with parse_file_triplet."""
    fname = tmp_path / "test_data.invalid"
    with open(fname, 'w') as f:
        f.write("Invalid format data")

    with pytest.raises(SystemExit):
        parse_file_triplet(str(fname))

def test_parse_file_triplet_normalize(tmp_path):
    """Test normalization functionality in parse_file_triplet."""
    fname = tmp_path / "test_data.csv"
    data = np.hstack([np.random.rand(10, 1), np.random.rand(10, 1), np.random.rand(10, 3)])
    columns = ["X", "Y"] + [f"V{i}" for i in range(data.shape[1] - 2)]
    pd.DataFrame(data, columns=columns).to_csv(fname, header=0, index=False, sep='\t')

    _, _, normalized_v = parse_file_triplet(str(fname), sep='\t', header=0, normalize=True)
    assert np.isclose(normalized_v.mean(), 0, atol=1e-6)
    assert np.isclose(normalized_v.std(), 1, atol=1e-6)
import os
import pytest
import numpy as np
from unittest import mock
from unittest.mock import ANY
from bayesgm.cli.cli import main
from bayesgm.utils.data_io import parse_file_triplet, save_data
from bayesgm.models import CausalBGM

def numpy_array_equal(array1, array2):
    return np.array_equal(array1, array2)

@pytest.fixture
def mock_data():
    """Fixture for creating mock data returned by parse_file_triplet."""
    x = np.random.rand(5, 1)
    y = np.random.rand(5, 1)
    v = np.random.rand(5, 10)
    return x, y, v

@mock.patch("bayesgm.cli.cli.parse_file_triplet")
@mock.patch("bayesgm.cli.cli.CausalBGM")
@mock.patch("bayesgm.cli.cli.save_data")
def test_main(mock_save_data, mock_causalbgm, mock_parse_file_triplet, tmp_path, mock_data):
    """Test the main function of cli.py."""
    # Mock the parse_file_triplet function to return mock data
    mock_parse_file_triplet.return_value = mock_data

    # Mock the CausalBGM object and its methods
    mock_model = mock.Mock()
    mock_model.predict.return_value = (np.array([1.0, 2.0, 1.0, 2.0, 3.0]), np.array([[0.5, 1.5], [1.5, 2.5], [0.8, 1.2], [1.8, 2.2], [2.9, 3.1]]))
    mock_model.save_dir = tmp_path
    mock_causalbgm.return_value = mock_model

    # Simulate command-line arguments (now uses subcommands)
    test_args = [
        "bayesgm",
        "causalbgm",
        "-o", str(tmp_path),
        "-i", "test_data.csv",
        "-t", "\t",
        "-B"  # binary_treatment
    ]
    with mock.patch("sys.argv", test_args):
        main()

    # Verify parse_file_triplet was called with the correct arguments
    mock_parse_file_triplet.assert_called_once_with("test_data.csv", sep="\t")

    # Verify the CausalBGM object was created with correct parameters
    mock_causalbgm.assert_called_once()
    assert "params" in mock_causalbgm.call_args[1]

    # Verify model methods were called
    mock_model.egm_init.assert_called_once()
    mock_model.fit.assert_called_once()
    mock_model.predict.assert_called_once()

    # Verify results were saved
    mock_save_data.assert_any_call(
        f"{mock_model.save_dir}/causal_effect_point_estimate.txt",
        ANY
    )
import numpy as np
import pytest
from unittest import mock
from unittest.mock import ANY

from bayesgm.cli.cli import main


@pytest.fixture
def mock_triplet_data():
    x = np.random.rand(8, 1).astype("float32")
    y = np.random.rand(8, 1).astype("float32")
    v = np.random.rand(8, 5).astype("float32")
    return x, y, v


@mock.patch("bayesgm.cli.cli.parse_file_triplet")
@mock.patch("bayesgm.cli.cli.CausalBGM")
@mock.patch("bayesgm.cli.cli.save_data")
def test_main_causalbgm_binary(
    mock_save_data,
    mock_causalbgm,
    mock_parse_file_triplet,
    tmp_path,
    mock_triplet_data,
):
    mock_parse_file_triplet.return_value = mock_triplet_data

    mock_model = mock.Mock()
    mock_model.predict.return_value = (
        np.array([1.0, 2.0, 3.0], dtype="float32"),
        np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]], dtype="float32"),
    )
    mock_model.save_dir = tmp_path
    mock_causalbgm.return_value = mock_model

    main([
        "causalbgm",
        "-o", str(tmp_path),
        "-i", "test_data.csv",
        "-t", "\t",
        "-B",
        "--burn_in", "7",
    ])

    mock_parse_file_triplet.assert_called_once_with("test_data.csv", sep="\t")
    mock_causalbgm.assert_called_once()
    assert "params" in mock_causalbgm.call_args.kwargs
    assert mock_causalbgm.call_args.kwargs["params"]["v_dim"] == mock_triplet_data[2].shape[1]

    mock_model.fit.assert_called_once_with(
        data=mock_triplet_data,
        epochs=100,
        epochs_per_eval=10,
        startoff=0,
        use_egm_init=True,
        egm_n_iter=30000,
        egm_batches_per_eval=500,
        verbose=1,
    )
    mock_model.predict.assert_called_once_with(
        data=mock_triplet_data,
        alpha=0.01,
        n_mcmc=3000,
        burn_in=7,
        q_sd=1.0,
    )
    mock_save_data.assert_any_call(f"{mock_model.save_dir}/causal_effect_point_estimate.txt", ANY)
    mock_save_data.assert_any_call(f"{mock_model.save_dir}/causal_effect_posterior_interval.txt", ANY)


@mock.patch("bayesgm.cli.cli.parse_file_triplet")
@mock.patch("bayesgm.cli.cli.CausalBGM")
@mock.patch("bayesgm.cli.cli.save_data")
def test_main_causalbgm_continuous(
    mock_save_data,
    mock_causalbgm,
    mock_parse_file_triplet,
    tmp_path,
    mock_triplet_data,
):
    mock_parse_file_triplet.return_value = mock_triplet_data

    mock_model = mock.Mock()
    mock_model.predict.return_value = (
        np.array([0.8, 1.1], dtype="float32"),
        np.array([[0.4, 1.2], [0.9, 1.3]], dtype="float32"),
    )
    mock_model.save_dir = tmp_path
    mock_causalbgm.return_value = mock_model

    main([
        "causalbgm",
        "-o", str(tmp_path),
        "-i", "test_data.csv",
        "-t", "\t",
        "--no-binary_treatment",
        "--x_values", "0.0", "1.5",
        "--burn_in", "9",
        "-F", "csv",
    ])

    mock_parse_file_triplet.assert_called_once_with("test_data.csv", sep="\t")
    mock_causalbgm.assert_called_once()
    assert mock_causalbgm.call_args.kwargs["params"]["binary_treatment"] is False

    mock_model.fit.assert_called_once_with(
        data=mock_triplet_data,
        epochs=100,
        epochs_per_eval=10,
        startoff=0,
        use_egm_init=True,
        egm_n_iter=30000,
        egm_batches_per_eval=500,
        verbose=1,
    )
    mock_model.predict.assert_called_once_with(
        data=mock_triplet_data,
        alpha=0.01,
        n_mcmc=3000,
        burn_in=9,
        x_values=[0.0, 1.5],
        q_sd=1.0,
    )
    mock_save_data.assert_any_call(f"{mock_model.save_dir}/causal_effect_point_estimate.csv", ANY)
    mock_save_data.assert_any_call(f"{mock_model.save_dir}/causal_effect_posterior_interval.csv", ANY)

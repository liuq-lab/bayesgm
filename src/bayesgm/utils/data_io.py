import numpy as np
import os
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler


def save_data(fname, data, delimiter='\t'):
    """
    Save the data to the specified path.

    Parameters:
    -----------
    fname : str
        The file name or path where the data will be saved.
    data : np.ndarray
        The data to save.
    delimiter : str, optional
        The delimiter for saving .txt or .csv files (default: '\t').

    Raises:
    -------
    ValueError
        If the file extension is not recognized.
    """
    if fname.endswith('.npy'):
        np.save(fname, data)
    elif fname.endswith('.txt') or fname.endswith('.csv'):
        np.savetxt(fname, data, fmt='%.6f', delimiter=delimiter)
    else:
        raise ValueError("Wrong saving format, please specify either .npy, .txt, or .csv")

def parse_file(path, sep='\t', header=0, normalize=True):
    """
    Parse an input data file and return a single data matrix.

    This is a general-purpose loader for the BGM model, where the input
    is a single data matrix (as opposed to the causal triplet format with
    treatment, outcome, and covariates).

    Parameters
    ----------
    path : str
        Path to the input file. Supported formats: .npz, .csv, .txt.
    sep : str, optional
        Separator for .csv or .txt files. Default is tab-delimited.
    header : int or None, optional
        Row number to use as column names in .csv files. Default is 0.
    normalize : bool, optional
        If True, the data will be normalized using ``StandardScaler``.

    Returns
    -------
    data : np.ndarray
        The data matrix with shape ``(n_samples, n_features)``, dtype float32.

    Examples
    --------
    >>> data = parse_file("data.csv", sep=',', normalize=True)
    >>> data = parse_file("data.npz", normalize=False)
    """
    assert os.path.exists(path), f"File not found: {path}"
    if path.endswith('npz'):
        loaded = np.load(path)
        # Support common key names: 'data', 'x', or the first key
        for key in ['data', 'x', 'X']:
            if key in loaded:
                data = loaded[key]
                break
        else:
            # Use the first available key
            first_key = list(loaded.keys())[0]
            data = loaded[first_key]
    elif path.endswith('csv'):
        data = pd.read_csv(path, header=header, sep=sep).values
    elif path.endswith('txt'):
        data = np.loadtxt(path, delimiter=sep)
    else:
        print('File format not recognized, please use .npz, .csv or .txt as input.')
        sys.exit()
    data = data.astype('float32')
    if normalize:
        data = StandardScaler().fit_transform(data)
    return data


def parse_file_triplet(path, sep='\t', header=0, normalize=True):
    """
    Parse an input file and extract the (treatment, outcome, covariates) triplet
    for CausalBGM model training or evaluation.

    Parameters
    ----------
    path : str
        Path to the input file. The file can be in .npz, .csv, or .txt format.
    sep : str, optional
        Separator used in .csv or .txt files. Defaults to tab-delimited format.
    header : int or None, optional
        Row number to use as column names in .csv files. Default is 0 (the first row). 
        Use `None` if the file does not have a header.
    normalize : bool, optional
        If True, the features in `v` will be normalized using `StandardScaler`.

    Returns
    -------
    data_x : np.ndarray
        The treatment variable(s) extracted from the file, reshaped to (-1, 1).
    data_y : np.ndarray
        The outcome variable(s) extracted from the file, reshaped to (-1, 1).
    data_v : np.ndarray
        Covariates extracted from the file. Normalized if `normalize=True`.

    Notes
    -----
    - Supported file formats:
        - `.npz`: Numpy compressed files with keys `x`, `y`, and `v`.
        - `.csv`: Comma-separated value files with treatment, outcome, and covariates as columns.
        - `.txt`: Tab- or other character-delimited text files with similar structure to .csv.
    - The input file must exist at the specified `path`.
    - The first column is assumed to be the treatment variable (`x`).
    - The second column is assumed to be the outcome variable (`y`).
    - Remaining columns are assumed to be covariates (`v`).

    Examples
    --------
    # Example for .csv input
    data_x, data_y, data_v = parse_file_triplet("data.csv", sep=',', header=0, normalize=True)
    
    # Example for .npz input
    data_x, data_y, data_v = parse_file_triplet("data.npz", normalize=False)
    """
    assert os.path.exists(path)
    if path[-3:] == 'npz':
        data = np.load(path)
        data_x, data_y, data_v = data['x'],data['y'],data['v']
    elif  path[-3:] == 'csv':
        data = pd.read_csv(path, header=0, sep=sep).values
        data_x = data[:,0].reshape(-1, 1).astype('float32')
        data_y = data[:,1].reshape(-1, 1).astype('float32')
        data_v = data[:,2:].astype('float32')
    elif path[-3:] == 'txt':
        data = np.loadtxt(path,delimiter=sep)
        data_x = data[:,0].reshape(-1, 1).astype('float32')
        data_y = data[:,1].reshape(-1, 1).astype('float32')
        data_v = data[:,2:].astype('float32')
    else:
        print('File format not recognized, please use .npz, .csv or .txt as input.')
        sys.exit()
    if normalize:
        data_v = StandardScaler().fit_transform(data_v)
    return data_x, data_y, data_v

# Installation

`bayesgm` can be installed via [pip](https://pypi.org/project/bayesgm/),
[conda](https://anaconda.org/conda-forge/bayesgm), and GitHub for Python users.

`bayesgm` includes both **BGM** and **CausalBGM** model families in one package. Model training can be faster with GPU, but it is not required.

## Prerequisites

1. Install conda via [miniconda](https://conda.pydata.org/miniconda.html) or
   [anaconda](https://www.anaconda.com/).
2. Create a new conda environment:

   ```bash
   conda create -n bayesgm-env python=3.9
   ```

3. Activate your environment:

   ```bash
   conda activate bayesgm-env
   ```

## Install with pip

Install from PyPI:

```bash
pip install bayesgm
```

If you get a `Permission denied` error, use:

```bash
pip install bayesgm --user
```

Install directly from GitHub source:

```bash
pip install git+https://github.com/liuq-lab/bayesgm.git
```

or install in editable mode:

```bash
git clone https://github.com/liuq-lab/bayesgm.git
cd bayesgm/src
pip install -e .
```

`-e` is short for `--editable`, which links the package to your local clone.

## Install with conda

1. Add `conda-forge` as highest-priority channel:

   ```bash
   conda config --add channels conda-forge
   ```

2. Enable strict channel priority:

   ```bash
   conda config --set channel_priority strict
   ```

3. Install:

   ```bash
   conda install -c conda-forge bayesgm
   ```

## Verify installation

```bash
python -c "import bayesgm; print(bayesgm.__version__)"
```

## Install R package for bayesgm (TODO)

bayesgm R package is built with
[reticulate](https://rstudio.github.io/reticulate/).

Install from CRAN:

```r
install.packages("bayesgm")
```

or from GitHub:

```r
devtools::install_github("liuq-lab/bayesgm", subdir = "r-package/bayesgm")
```



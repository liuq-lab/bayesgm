# Installation

`bayesgm` can be installed via [pip](https://pypi.org/project/bayesgm/),
[conda](https://anaconda.org/conda-forge/bayesgm), and GitHub for Python users.

`bayesgm` includes both **BGM** and **CausalBGM** model families in one package.

## Prerequisites

### pip prerequisites

1. Install [Python](https://www.python.org/downloads/). We recommend Python >= 3.9 and using
   [venv](https://docs.python.org/3/library/venv.html) or
   [pyenv](https://github.com/pyenv/pyenv) for virtual-environment management.
2. Create a virtual environment:

   ```bash
   python3 -m venv <venv_path>
   ```

3. Activate the environment:

   ```bash
   source <venv_path>/bin/activate
   ```

### conda prerequisites

1. Install conda via [miniconda](https://conda.pydata.org/miniconda.html) or
   [anaconda](https://www.anaconda.com/).
2. Create a new conda environment:

   ```bash
   conda create -n bayesgm-env python=3.10
   ```

3. Activate your environment:

   ```bash
   conda activate bayesgm-env
   ```

### GPU prerequisites (optional)

Training is faster with GPU acceleration. If you plan to use GPU, configure CUDA and cuDNN
before installing dependencies.

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

## Install R package for CausalBGM users (optional)

For R workflows, you can use
[RcausalBGM](https://cran.r-project.org/web/packages/RcausalBGM/index.html),
which is built with
[reticulate](https://rstudio.github.io/reticulate/).

Install from CRAN:

```r
install.packages("RcausalBGM")
```

or from GitHub:

```r
devtools::install_github("SUwonglab/CausalBGM", subdir = "r-package/RcausalBGM")
```

## Verify installation

```bash
python -c "import bayesgm; print(bayesgm.__version__)"
```

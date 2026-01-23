[![PyPI](https://img.shields.io/pypi/v/bayesgm)](https://pypi.org/project/bayesgm/)
[![Anaconda](https://anaconda.org/conda-forge/causalegm/badges/version.svg)](https://anaconda.org/conda-forge/causalegm)
[![Travis (.org)](https://app.travis-ci.com/kimmo1019/CausalEGM.svg?branch=main)](https://app.travis-ci.com/github/kimmo1019/CausalEGM)
[![All Platforms](https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/causalegm-feedstock?branchName=main)](https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=18625&branchName=main)
[![Documentation Status](https://readthedocs.org/projects/causalbgm/badge/?version=latest)](https://causalbgm.readthedocs.io)


# bayesgm: A Versatile Bayesian Generative Modeling Framework

**bayesgm** is a Python package providing a **unified Bayesian generative modeling (BGM) framework** for representation learning, uncertainty quantification, and downstream tasks such as **causal inference** in complex, high-dimensional data.

The framework is built upon **Bayesian principles combined with modern deep generative models**, enabling flexible modeling of latent structures, complex dependencies, and principled uncertainty estimation.

**BGM** is the foundational component of `bayesgm`, designed for **general-purpose Bayesian generative modeling**. It also serves as the **modeling backbone** upon which task-specific extensions (e.g., causal inference, Bayesian inference) are built.

---

## Installation

See detailed installation instructions in our [website](https://causalbgm.readthedocs.io/en/latest/installation.html). Briefly, **bayesgm** Python package can be installed via 

```bash
pip install bayesgm
```

## Core Components

### 1️⃣ BGM: A Bayesian Generative Modeling Approach for Arbitrary Conditional Inference

BGM is the foundational module in `bayesgm` for Bayesian generative modeling and arbitrary conditional inference in high-dimensional settings.

With a trained BGM model, you can::

- Train once, infer anywhere: condition on any subset of observed dimensions without retraining.
- Obtain Bayesian posterior samples of missing/unobserved parts. 
- Produce uncertainty quantification (e.g., posterior predictive intervals) in addition to point estimates.
- Applications including conditional prediction/generation, missing data imputation.


#### Usage

A detailed Python tutorial can be found at our [website](https://causalbgm.readthedocs.io/en/latest/tutorial_py.html).

##### Example Usage of BGM

```python
import yaml
import numpy as np
import bayesgm
from bayesgm.models import BayesGM
from bayesgm.utils import simulate_z_hetero
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open('src/configs/Sim_heteroskedastic.yaml', 'r'))
X, Y = simulate_z_hetero(n=20000, k=10, d=params['x_dim']-1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=123)
data = np.c_[X_train, Y_train].astype('float32')

# Instantiate a BGM model
model = BayesGM(params=params, random_seed=None)

# Perform Encoding Generative Modeling (EGM) initialization (optional but recommended)
model.egm_init(data=data, n_iter=30000, batches_per_eval=5000, verbose=1)

# Train the BGM model with an iterative updating algorithm
model.fit(data=data, epochs=500, epochs_per_eval=10, verbose=1)

# Provide both point estimate and posterior interval (uncertainty) using the trained BGM model
ind_x1 = list(range(params['x_dim']-1))
data_x_pred, pred_interval = model.predict(data=X_test,ind_x1=ind_x1,alpha=0.05,bs=100,seed=42)
X_test_pred = data_x_pred[:,:,-1]
X_test_pred_mean = np.mean(X_test_pred, axis=0)
```
---

### 2️⃣ CausalBGM: An AI-powered Bayesian Generative Modeling Approach for Causal Inference in Observational Studies


### <a href='https://causalbgm.readthedocs.io/'><img src='https://raw.githubusercontent.com/SUwonglab/CausalBGM/main/docs/source/logo.png' align="left" height="60" /></a> 
**CausalBGM** is a specialized module built **on top of BGM** for causal inference from observational data.

<a href='https://causalbgm.readthedocs.io/'><img align="left" src="https://github.com/SUwonglab/CausalBGM/blob/main/model.png" width="500">

CausalBGM adopts a Bayesian iterative approach to update the model parameters and the posterior distribution of latent features until convergence. This framework leverages the power of AI to capture complex dependencies among variables while adhering to the Bayesian principles.

CausalBGM was developed with Python3.9, TensorFlow2.10, and TensorFlow Probability. Now [Python PyPI package]((https://pypi.org/project/CausalBGM/)) for CausalBGM is available. Besides, we provide a console program to run CausalBGM directly. For more information, checkout the [Document](https://causalbgm.readthedocs.io/).

#### CausalBGM Main Applications

- Point estimate of  ATE, ITE, ADRF, CATE.

- Posterior interval estimate of ATE, ITE, ADRF, CATE with user-specific significant level α (alpha).

#### Usage

A detailed Python tutorial can be found at our [website](https://causalbgm.readthedocs.io/en/latest/tutorial_py.html). The source Python notebook for the detailed tutorial is provided at [here](https://github.com/SUwonglab/CausalBGM/blob/main/docs/source/tutorial_py.ipynb).

##### Example Usage of CausalBGM

```python
import yaml
import numpy as np
import bayesgm
from bayesgm.models import CausalBGM
from bayesgm.datasets import Sim_Hirano_Imbens_sampler

params = yaml.safe_load(open('src/configs/Sim_Hirano_Imbens.yaml', 'r'))
x, y, v = Sim_Hirano_Imbens_sampler(N=20000, v_dim=200).load_all()

# Instantiate a CausalBGM model
model = CausalBGM(params=params, random_seed=None)

# Perform Encoding Generative Modeling (EGM) initialization (optional but recommended)
model.egm_init(data=(x, y, v), n_iter=30000, batches_per_eval=500, verbose=1)

# Train the CausalBGM model with an iterative updating algorithm
model.fit(data=(x, y, v), epochs=100, epochs_per_eval=10, verbose=1)

# Provide both point estimate and posterior interval (uncertainty) using the trained CausalBGM model
causal_pre, pos_intervals = model.predict(
  data=(x, y, v), alpha=0.01, n_mcmc=3000, x_values=np.linspace(0, 3, 20), q_sd=1.0
)
```

#### Datasets

`bayesgm` package provides several built-in simulation datasets from `bayesgm.datasets`.

For semi-synthetic dataset, users need to create a `CausalBGM/data` folder and uncompress the dataset in the `CausalBGM/data` folder.

- [Twin dataset](https://www.nber.org/research/data/linked-birthinfant-death-cohort-data). Google Drive download [link](https://drive.google.com/file/d/1fKCb-SHNKLsx17fezaHrR2j29T3uD0C2/view?usp=sharing).

- [ACIC 2018 datasets](https://www.synapse.org/#!Synapse:syn11294478/wiki/494269). Google Drive download [link](https://drive.google.com/file/d/1qsYTP8NGh82nFNr736xrMsJxP73gN9OG/view?usp=sharing).
  

## Main References

- Qiao Liu and Wing Hung Wong. [An AI-powered Bayesian generative modeling approach for causal inference in observational studies](https://arxiv.org/abs/2501.00755) [J]. arXiv preprint arXiv:2501.00755, 2025.

- Qiao Liu, Zhongren Chen, and Wing Hung Wong. [An encoding generative modeling approach to dimension reduction and covariate adjustment in causal inference with observational studies](https://www.pnas.org/doi/10.1073/pnas.2322376121) [J]. PNAS, 121 (23) e2322376121, 2024.

## Support

Found a bug or would like to see a feature implemented? Feel free to submit an [issue](https://github.com/SUwonglab/CausalBGM/issues/new/choose). 

Have a question or would like to start a new discussion? You can also always send us an [e-mail](mailto:liuqiao@stanford.edu?subject=[GitHub]%20CausalBGM%20project). 

Your help to improve CausalBGM is highly appreciated! For further information visit [website](https://causalbgm.readthedocs.io/).


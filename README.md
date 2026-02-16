<a href="https://scvi-tools.org/">
  <img
    src="https://github.com/liuq-lab/bayesgm/blob/main/docs/source/logo.png"
    width="400"
    alt="bayesgm"
  >
</a>

[![PyPI](https://img.shields.io/pypi/v/bayesgm)](https://pypi.org/project/bayesgm/)
[![Anaconda](https://anaconda.org/conda-forge/causalegm/badges/version.svg)](https://anaconda.org/conda-forge/causalegm)
[![Travis (.org)](https://app.travis-ci.com/kimmo1019/CausalEGM.svg?branch=main)](https://app.travis-ci.com/github/kimmo1019/CausalEGM)
[![All Platforms](https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/causalegm-feedstock?branchName=main)](https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=18625&branchName=main)
[![Documentation Status](https://readthedocs.org/projects/bayesgm/badge/?version=latest)](https://bayesgm.readthedocs.io)


# bayesgm: A toolkit for AI-driven Bayesian Generative Modeling

**bayesgm** is a toolkit  providing a AI-driven Bayesian generative modeling framework for various Bayesian inference tasks in complex, high-dimensional data.

The framework is built upon Bayesian principles combined with modern AI models, enabling flexible modeling of complex dependencies with principled uncertainty estimation.

Currently, the **bayesgm** package includes two model families:

- **BGM**: Bayesian generative modeling for arbitrary conditional inference (foundational model).
- **CausalBGM**: Bayesian generative modeling for causal effect estimation.

---

## Utilities

**bayesgm** toolkit can be used for a wide range of tasks based on Bayesian principle with **Uncertainty Quantification**, including:

- Data Generation
- Bayesian Posterior Prediction
- Missing Data Imputation
- Counterfactual Prediction
- Causal Effect Estimation

We provide an overview in the [user guide](https://bayesgm.readthedocs.io/en/latest/getting-started/user_guide.html). All model implementations have a
high-level API that supports model instantiation, training, inference, save/load functions, etc.

## Installation

See detailed in our [Installation Page](https://bayesgm.readthedocs.io/en/latest/getting-started/installation.html). Briefly, **bayesgm** Python package can be installed via 

```bash
pip install bayesgm
```
## Resources

- Tutorials, API reference, and installation guides are available in the [Documentation](https://bayesgm.readthedocs.io/).

## Main References

If you use `bayesgm` tool in your work, please consider citing the corresponding publications:

- Qiao Liu and Wing Hung Wong, [A Bayesian Generative Modeling Approach for Arbitrary Conditional Inference](https://arxiv.org/abs/2601.05355) [J]. arXiv preprint arXiv:2601.05355, 2026

- Qiao Liu and Wing Hung Wong. [An AI-powered Bayesian generative modeling approach for causal inference in observational studies](https://arxiv.org/abs/2501.00755) [J]. arXiv preprint arXiv:2501.00755, 2025 (minor revision at JASA).

- Qiao Liu, Zhongren Chen, and Wing Hung Wong. [An encoding generative modeling approach to dimension reduction and covariate adjustment in causal inference with observational studies](https://www.pnas.org/doi/10.1073/pnas.2322376121) [J]. PNAS, 121 (23) e2322376121, 2024.

## Support

Found a bug or would like to see a feature implemented? Feel free to submit an [issue](https://github.com/liuq-lab/bayesgm/issues/new/choose). 

Have a question or would like to start a new discussion? You can also send me an [e-mail](mailto:qiao.liu@yale.edu?subject=[GitHub]%20bayesgm%20project). 


# User guide

`bayesgm` is a toolkit providing a AI-driven Bayesian generative modeling framework for various Bayesian inference tasks in complex, high-dimensional data.

The figure below illustrates the versatility of `bayesgm`, spanning dimensional reduction, data generation, Bayesian posterior inference, missing-data imputation, causal effect estimation, and counterfactual prediction:

![bayesgm versatility](../bayesgm.png)

## Which model should I use?

Use **BGM** family if your goal is:

- conditional prediction/generation
- missing-data imputation
- dimension reduction

Use **CausalBGM** family if your goal is:

- counterfactual prediction
- ATE estimation
- ITE estimation

## Package overview

All models are installed from the same `bayesgm` package:

```bash
pip install bayesgm
```

Core namespaces:

- `bayesgm.models` for model classes (`BGM`, `CausalBGM`, etc.)
- `bayesgm.datasets` for built-in simulation/semi-synthetic samplers
- `bayesgm.utils` for helpers and data IO

Next steps:

1. Follow the **Installation** page in this section.
2. Open the **BGM** or **CausalBGM** section in the sidebar.
3. Start from the model quickstart block, then continue to tutorials.

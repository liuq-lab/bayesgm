Datasets
========

Causal Inference Samplers
-------------------------

These samplers generate or load causal inference datasets with treatment (x),
outcome (y), and covariates (v). They are used with the **CausalBGM** model
family.

.. currentmodule:: bayesgm.datasets.base_sampler

.. autoclass:: Base_sampler
   :members: next_batch, load_all

.. currentmodule:: bayesgm.datasets.causal_samplers

.. autoclass:: Sim_Hirano_Imbens_sampler
   :members:
   :show-inheritance:

.. autoclass:: Sim_Sun_sampler
   :members:
   :show-inheritance:

.. autoclass:: Sim_Colangelo_sampler
   :members:
   :show-inheritance:

.. autoclass:: Semi_Twins_sampler
   :members:
   :show-inheritance:

.. autoclass:: Semi_acic_sampler
   :members:
   :show-inheritance:


Prior / Distribution Samplers
-----------------------------

These samplers generate data from known distributions. They are used as
latent-space priors or benchmark datasets for the **BGM** model family.

.. currentmodule:: bayesgm.datasets.prior_samplers

.. autoclass:: Gaussian_sampler
   :members: train, get_batch, load_all

.. autoclass:: GMM_indep_sampler
   :members: train, get_density, load_all

.. autoclass:: Swiss_roll_sampler
   :members: train, get_density, load_all


Simulation Functions
--------------------

Functions for generating synthetic datasets for **BGM** experiments.

.. currentmodule:: bayesgm.datasets.simulators

.. autofunction:: simulate_regression

.. autofunction:: simulate_low_rank_data

.. autofunction:: simulate_heteroskedastic_data

.. autofunction:: simulate_z_hetero


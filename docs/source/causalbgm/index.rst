CausalBGM - An AI-powered Bayesian generative modeling approach for causal inference in observational studies
=============================================================================================================

.. image:: https://raw.githubusercontent.com/SUwonglab/CausalBGM/main/model.png
   :width: 800px
   :align: left

**CausalBGM** is an innovative Bayesian generative modeling framework tailored for
causal inference in observational studies with **high-dimensional covariates**
and **large-scale datasets**.

It addresses key challenges by leveraging Bayesian principles and advanced AI techniques to estimate average treatment effects (ATEs) and individual treatment effects (ITEs) with robust uncertainty quantification.

CausalBGM applicability
^^^^^^^^^^^^^^^^^^^^^^^

- Point estimates of counterfactual outcomes, ATE, ITE, ADRF, and CATE.
- Posterior interval estimates for counterfactual outcomes, ATE, ITE, ADRF, and CATE.
- Support for both continuous and binary treatment settings.


Method highlights
^^^^^^^^^^^^^^^^^

#. CausalBGM combines Bayesian causal inference with AI techniques for principled and scalable causal effect estimation.
#. CausalBGM adopts an encoding generative modeling (EGM) initialization strategy for stable training.
#. CausalBGM outperforms leading causal inference methods in various settings.


Quickstart
^^^^^^^^^^

.. code-block:: python

   import yaml
   import numpy as np
   from bayesgm.models import CausalBGM
   from bayesgm.datasets import Sim_Hirano_Imbens_sampler

   params = yaml.safe_load(open("src/configs/Sim_Hirano_Imbens.yaml", "r"))
   x, y, v = Sim_Hirano_Imbens_sampler(N=20000, v_dim=200).load_all()

   # Instantiate a CausalBGM model
   model = CausalBGM(params=params, random_seed=None)

   # Train the CausalBGM model with EGM initialization and iterative updating algorithm
   model.fit(data=(x, y, v), epochs=200, epochs_per_eval=10, use_egm_init=True, egm_n_iter=30000, egm_batches_per_eval=500, verbose=1)

   # Make predictions using the trained CausalBGM model
   causal_pre, pos_intervals = model.predict(
       data=(x, y, v),
       alpha=0.01,
       n_mcmc=3000,
       x_values=np.linspace(0, 3, 20),
       q_sd=1.0,
   )

Main references
^^^^^^^^^^^^^^^

- Qiao Liu and Wing Hung Wong (2025), An AI-powered Bayesian generative modeling approach for causal inference in observational studies.
  `JASA (in press) <https://arxiv.org/abs/2501.00755>`__.

- Qiao Liu, Zhongren Chen, and Wing Hung Wong (2024), An encoding generative modeling approach to dimension reduction and covariate adjustment in causal inference with observational studies.
  `PNAS <https://www.pnas.org/doi/10.1073/pnas.2322376121>`__.

Tutorials
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   Tutorial for Python users <tutorial_py>
   Tutorial for R users <tutorial_r>

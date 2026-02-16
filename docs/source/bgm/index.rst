BGM - A Bayesian Generative Modeling Approach for Arbitrary Conditional Inference
===================================================================================

**BGM** is the foundational model family in `bayesgm` for arbitrary conditional
inference in high-dimensional settings.

With a trained BGM model, users can perform inference under many observed/missing
patterns without retraining from scratch.

BGM wide applicability
^^^^^^^^^^^^^^^^^^^^^^

- Train once, infer under multiple conditioning patterns.
- Obtain posterior samples for missing or unobserved components.
- Produce posterior interval estimates in addition to point predictions.
- Support conditional prediction, generation, and imputation tasks.

Quickstart
^^^^^^^^^^

.. code-block:: python

   import yaml
   import numpy as np
   from sklearn.model_selection import train_test_split
   from bayesgm.models import BGM
   from bayesgm.datasets import simulate_z_hetero

   params = yaml.safe_load(open("src/configs/Sim_heteroskedastic.yaml", "r"))
   X, Y = simulate_z_hetero(n=20000, k=10, d=params["x_dim"] - 1)
   X_train, X_test, Y_train, _ = train_test_split(X, Y, test_size=0.1, random_state=123)
   data = np.c_[X_train, Y_train].astype("float32")

   model = BGM(params=params, random_seed=None)
   model.egm_init(data=data, n_iter=30000, batches_per_eval=5000, verbose=1)
   model.fit(data=data, epochs=500, epochs_per_eval=10, verbose=1)

   ind_x1 = list(range(params["x_dim"] - 1))
   data_x_pred, pred_interval = model.predict(
       data=X_test, ind_x1=ind_x1, alpha=0.05, bs=100, seed=42
   )

Method highlights
^^^^^^^^^^^^^^^^^

#. Bayesian generative modeling for principled uncertainty quantification.
#. Iterative updating for posterior learning of latent features and parameters.
#. Flexible neural architectures for complex nonlinear dependencies.

Main reference
^^^^^^^^^^^^^^

- Qiao Liu and Wing Hung Wong (2026), A Bayesian Generative Modeling Approach for Arbitrary Conditional Inference.
  `arXiv <https://arxiv.org/abs/2601.05355>`__.

Tutorials
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   Tutorial for Python users <tutorial_py>

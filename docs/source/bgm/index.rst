BGM - A Bayesian Generative Modeling Approach for Arbitrary Conditional Inference
===================================================================================

**BGM** is the foundational model family in `bayesgm` for arbitrary conditional
inference in high-dimensional settings.

With a trained BGM model, users can perform inference under any observed/missing
patterns without retraining from scratch.

BGM applicability
^^^^^^^^^^^^^^^^^

- Train once, infer under multiple conditioning patterns.
- Obtain posterior samples for missing or unobserved components.
- Produce posterior interval estimates in addition to point predictions.
- Support conditional prediction with prediction intervals (e.g., conformal prediction).

Method highlights
^^^^^^^^^^^^^^^^^

#. BGM combines Bayesian principles with modern AI techniques for arbitrary conditional inference.
#. BGM provides significant flexibility and scalability to efficiently handle large, high-dimensional datasets.
#. BGM outperforms leading conformal prediction methods inpredictive accuracy and calibrated uncertainty quantification.

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
   data_train = np.c_[X_train, Y_train].astype("float32")

   # Instantiate a BGM model
   model = BGM(params=params, random_seed=None)

   # Train the BGM model with EGM initialization and iterative updating algorithm
   model.fit(data=data_train, epochs=200, epochs_per_eval=10, use_egm_init=True, egm_n_iter=50000, egm_batches_per_eval=500, verbose=1)

   # Prepare test data with missing values
   data_test = np.hstack([X_test, np.full((X_test.shape[0], 1), np.nan)])

   # Make predictions using the trained BGM model
   data_x_pred, pred_interval = model.predict(
       data=data_test, alpha=0.05, n_mcmc=5000, step_size=0.01, seed=42
   )


Main reference
^^^^^^^^^^^^^^

- Qiao Liu and Wing Hung Wong (2026), A Bayesian Generative Modeling Approach for Arbitrary Conditional Inference.
  `arXiv <https://arxiv.org/abs/2601.05355>`__.

Tutorials
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   Tutorial for Python users <tutorial_py>

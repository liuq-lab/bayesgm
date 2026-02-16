# BGM Python Tutorial

This page gives a concise end-to-end example using simulated heteroskedastic data.

```python
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from bayesgm.models import BGM
from bayesgm.utils import simulate_z_hetero

params = yaml.safe_load(open("src/configs/Sim_heteroskedastic.yaml", "r"))
X, Y = simulate_z_hetero(n=20000, k=10, d=params["x_dim"] - 1)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=123
)
train_data = np.c_[X_train, Y_train].astype("float32")

model = BGM(params=params, random_seed=None)
model.egm_init(data=train_data, n_iter=30000, batches_per_eval=5000, verbose=1)
model.fit(data=train_data, epochs=500, epochs_per_eval=10, verbose=1)

ind_x1 = list(range(params["x_dim"] - 1))
data_x_pred, pred_interval = model.predict(
    data=X_test, ind_x1=ind_x1, alpha=0.05, bs=100, seed=42
)
```

Outputs:

- `data_x_pred`: posterior predictive samples.
- `pred_interval`: posterior interval estimates under selected confidence level.

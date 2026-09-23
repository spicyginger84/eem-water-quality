"""Classical regression model factories."""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def make_model(name, seed=42, n_jobs=1, log_target=False):
    """Construct one unfitted classical estimator."""
    if name == "linear":
        model = LinearRegression(n_jobs=n_jobs)
    elif name == "tree":
        model = DecisionTreeRegressor(random_state=seed)
    elif name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ImportError("Install the 'boosting' extra to run xgboost") from error
        model = XGBRegressor(random_state=seed, n_jobs=n_jobs)
    elif name == "svr":
        model = TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")
            ),
            transformer=StandardScaler(),
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    if log_target:
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return model


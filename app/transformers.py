import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):

    def __init__(self, drop_columns=None):
        self.drop_columns = drop_columns if drop_columns is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # 1. Balance por producto
        X["BalancePerProduct"] = (
            X["Balance"] /
            X["NumOfProducts"].replace(0, np.nan)
        )

        X["BalancePerProduct"] = X["BalancePerProduct"].fillna(0)

        # 4. Cliente comprometido
        X["EngagedCustomer"] = (
            (X["IsActiveMember"] == 1) &
            (X["NumOfProducts"] >= 2)
        ).astype(int)

        # 5. Antigüedad relativa
        X["TenureAgeRatio"] = (
            X["Tenure"] /
            X["Age"]
        )

        X.drop(
            columns=self.drop_columns,
            errors="ignore",
            inplace=True
        )

        return X
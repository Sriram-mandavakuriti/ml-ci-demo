"""Load and preprocess the UCI Mushroom dataset."""

import pandas as pd
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42

DROP_COLUMNS = ["veil-type"]


def load_raw():
    """Download and load the original mushroom dataset."""
    ds = fetch_ucirepo(id=73)

    X = ds.data.features.copy()
    y = ds.data.targets.iloc[:, 0].copy()

    return X, y


def preprocess(X, y):
    """Clean and encode mushroom dataset."""

    X = X.drop(columns=[c for c in DROP_COLUMNS if c in X.columns])

    # Replace missing values
    X = X.fillna("unknown")

    # Convert categorical features into numerical features
    X = pd.get_dummies(X)

    # Encode target: 1 = poisonous, 0 = edible
    y = (y == "p").astype(int)

    return X, y


def load_mushroom():
    """Load and preprocess the complete dataset."""
    return preprocess(*load_raw())
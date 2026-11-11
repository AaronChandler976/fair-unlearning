
import numpy as np
import pandas as pd
import torch

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_hsls(
    data_path="data/raw/hsls/hsls_16_student_v1_0.csv",
    test_size=0.2,
    random_state=42,
):
    """
    Load the HSLS 2016 student dataset.

    Prediction target:
        X4EVRATNDCLG: Ever attended college (0 or 1).

    Protected attribute:
        X1SEX: Male (0), Female (1).

    Predictors:
        X3TGPATOT: Overall high-school GPA.
        X1FAMINCOME: Family income category.

    Returns a dictionary compatible with the Adult and COMPAS loaders.
    """

    columns = [
        "X4EVRATNDCLG",
        "X4EVRATNDCLG_IM",
        "X1SEX",
        "X1FAMINCOME",
        "X3TGPATOT",
    ]

    df = pd.read_csv(data_path, usecols=columns)

    # Convert relevant columns to numeric.
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Keep valid, non-imputed college attendance outcomes.
    df = df[
        df["X4EVRATNDCLG"].isin([0, 1])
        & (df["X4EVRATNDCLG_IM"] == 0)
    ].copy()

    # Keep records with a valid protected attribute.
    df = df[df["X1SEX"].isin([1, 2])].copy()

    # Encode sex: male = 0, female = 1.
    sensitive = (df["X1SEX"] == 2).astype(int)

    # Binary prediction target.
    y = df["X4EVRATNDCLG"].astype(int)

    # Prepare predictors.
    X = df[["X3TGPATOT", "X1FAMINCOME"]].copy()

    # Special negative GPA codes represent missing or unavailable values.
    X["X3TGPATOT"] = X["X3TGPATOT"].where(
        X["X3TGPATOT"] >= 0, np.nan
    )

    # Treat income as categorical, not continuous.
    # Negative codes become missing values.
    X["X1FAMINCOME"] = X["X1FAMINCOME"].where(
        X["X1FAMINCOME"] >= 1, np.nan
    )
    X["X1FAMINCOME"] = X["X1FAMINCOME"].map(
        lambda value: str(int(value)) if pd.notna(value) else np.nan
    )

    # Split before fitting preprocessing to avoid data leakage.
    stratify_labels = (
        y.astype(str) + "_" + sensitive.astype(str)
    )

    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = (
        train_test_split(
            X,
            y,
            sensitive,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )
    )

    # Numeric GPA: median imputation, missing indicator, standardization.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True,
        )),
        ("scaler", StandardScaler()),
    ])

    # Income: missing category and one-hot encoding.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Missing",
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, ["X3TGPATOT"]),
        ("categorical", categorical_pipeline, ["X1FAMINCOME"]),
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()

    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    return {
        "X_train": torch.tensor(X_train, dtype=torch.float64),
        "y_train": torch.tensor(
            y_train.to_numpy(), dtype=torch.float64
        ),
        "sensitive_train": torch.tensor(
            sensitive_train.to_numpy(), dtype=torch.long
        ),
        "X_test": torch.tensor(X_test, dtype=torch.float64),
        "y_test": torch.tensor(
            y_test.to_numpy(), dtype=torch.float64
        ),
        "sensitive_test": torch.tensor(
            sensitive_test.to_numpy(), dtype=torch.long
        ),
        "feature_names": preprocessor.get_feature_names_out(),
        "preprocessor": preprocessor,
    }

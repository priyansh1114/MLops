from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "heart.csv"
MODEL_DIR = ROOT / "artifacts"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

CATEGORICAL_COLUMNS = ["sex", "cp", "restecg", "slope", "thal"]
NUMERIC_COLUMNS = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS]


def download_dataset(data_path: Path = DATA_PATH) -> Path:
    """Download the Cleveland heart disease dataset if it is missing."""
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        return data_path

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    urllib.request.urlretrieve(url, data_path)
    return data_path


def load_dataset() -> pd.DataFrame:
    """Load and clean the dataset into a tabular format ready for modeling."""
    file_path = download_dataset()
    df = pd.read_csv(
        file_path,
        header=None,
        names=[*FEATURE_COLUMNS, "target"],
        na_values="?",
    )
    df = df.copy()
    df["target"] = pd.to_numeric(df["target"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    for column in FEATURE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["sex"] = df["sex"].astype("Int64")
    df["cp"] = df["cp"].astype("Int64")
    df["restecg"] = df["restecg"].astype("Int64")
    df["slope"] = df["slope"].astype("Int64")
    df["thal"] = df["thal"].astype("Int64")
    df["ca"] = df["ca"].astype("Int64")
    df["target"] = df["target"].clip(lower=0, upper=1)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline for numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    return preprocessor

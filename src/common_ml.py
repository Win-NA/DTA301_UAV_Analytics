import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)


from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model, filename):
    model_path = MODEL_DIR / filename

    joblib.dump(model, model_path)

    print(f"Model saved: {model_path}")

    return model_path

def load_dataset():
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "data/processed/SurveilDrone_Net23_cleaned.csv",
        base_dir.parent / "data/processed/SurveilDrone_Net23_cleaned.csv",
        base_dir / "data/SurveilDrone-Net23.csv",
        base_dir.parent / "data/raw/SurveilDrone-Net23.csv",
        base_dir.parent / "data/SurveilDrone-Net23.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            frame.columns = [str(c).strip() for c in frame.columns]
            return frame, path
    raise FileNotFoundError(
        "Dataset CSV not found. Update load_dataset() with the actual CSV path."
    )


def make_preprocessor(numeric_features, categorical_features):
    transformers = []
    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def evaluate_classifier(model, X_test, y_test):
    pred = model.predict(X_test)
    result = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, zero_division=0)),
    }
    if hasattr(model, "predict_proba") and y_test.nunique() == 2:
        probabilities = model.predict_proba(X_test)
        classes = list(model.classes_)
        result["auc_roc"] = (
            float(roc_auc_score(y_test, probabilities[:, classes.index(1)]))
            if 1 in classes else None
        )
    else:
        result["auc_roc"] = None
    return result


def train_classifier(X, y, numeric_features, categorical_features, estimator):
    y = pd.to_numeric(y, errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid].copy(), y.loc[valid].astype(int)
    if y.nunique() != 2:
        raise ValueError(
            f"Need both binary classes 0 and 1; found {sorted(y.unique())}."
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
        ("model", estimator),
    ])
    model.fit(X_train, y_train)
    return model, evaluate_classifier(model, X_test, y_test), len(X), len(X_test)

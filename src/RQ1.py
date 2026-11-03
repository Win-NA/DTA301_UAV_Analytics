
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


class RQ1Service:

    FEATURES = [
        "battery_level_pct",
        "wind_speed_mps",
        "proximity_to_restricted_zone_m",
        "ambient_temp_C"
    ]

    TARGET = "unstable_flight"

    def __init__(self, df, model_path="models/rq1_classifier.pkl"):

        self.df = df.copy()

        # Resolve model path relative to project root
        project_root = Path(__file__).resolve().parent.parent

        self.model_path = Path(model_path)

        if not self.model_path.is_absolute():
            self.model_path = project_root / self.model_path

        # Automatically create models directory
        self.model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Generate target if missing
        if self.TARGET not in self.df.columns:
            self.df[self.TARGET] = (
                (self.df["battery_level_pct"] < 20)
                | (self.df["wind_speed_mps"] > 8.0)
                | (self.df["proximity_to_restricted_zone_m"] < 50)
            ).astype(int)

        # Prepare dataset
        self.X = self.df[self.FEATURES]
        self.y = self.df[self.TARGET].astype(int)

        self.model = None
        self.metrics = {}

        self.train()

    def train(self):

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )

        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)

        probabilities = self.model.predict_proba(X_test)

        positive_index = list(self.model.classes_).index(1)

        positive_probabilities = probabilities[:, positive_index]

        self.metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "f1_score": float(
                f1_score(y_test, predictions)
            ),
            "auc_roc": float(
                roc_auc_score(y_test, positive_probabilities)
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions
            ).tolist()
        }

        print("\n========== RQ1 ==========")
        print(self.metrics)
        print(classification_report(y_test, predictions))

        # Save trained model
        joblib.dump(self.model, self.model_path)

        print(f"RQ1 model saved to: {self.model_path}")

    def predict(self, current):

        input_data = pd.DataFrame([{
            feature: current.get(feature, 0)
            for feature in self.FEATURES
        }])

        prediction = int(
            self.model.predict(input_data)[0]
        )

        probabilities = self.model.predict_proba(
            input_data
        )[0]

        positive_index = list(
            self.model.classes_
        ).index(1)

        return {
            "is_unstable": prediction,
            "risk_probability": float(
                probabilities[positive_index]
            )
        }

    def get_metrics(self):
        return self.metrics

    def get_feature_importance(self):

        return {
            feature: float(importance)
            for feature, importance in zip(
                self.FEATURES,
                self.model.feature_importances_
            )
        }

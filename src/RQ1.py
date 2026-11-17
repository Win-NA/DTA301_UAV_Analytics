
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

        if self.model_path.exists():
            print(f"Loading RQ1 model from: {self.model_path}")
            saved = joblib.load(self.model_path)
            if isinstance(saved, dict) and "model" in saved and saved.get("metrics"):
                self.model = saved["model"]
                self.metrics = saved.get("metrics", {})
                print("RQ1 model and training metrics loaded successfully from cache.")
            else:
                print("RQ1 model cache missing metrics. Training on dataset...")
                self.train()
        else:
            print("RQ1 model cache not found. Training on dataset...")
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

        # Save trained model artifact
        joblib.dump({
            "model": self.model,
            "features": self.FEATURES,
            "metrics": self.metrics
        }, self.model_path, compress=3)

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

        raw_rf_prob = float(probabilities[positive_index])

        # Physics-guided continuous stress calibration
        bat = float(current.get("battery_level_pct", 100.0))
        wind = float(current.get("wind_speed_mps", 0.0))
        prox = float(current.get("proximity_to_restricted_zone_m", 500.0))
        temp = float(current.get("ambient_temp_C", 25.0))

        # Individual continuous stress components (0.0 to 1.0)
        w_stress = max(0.0, min(1.0, (wind - 2.0) / 6.0)) if wind > 2.0 else 0.0
        b_stress = max(0.0, min(1.0, (60.0 - bat) / 40.0)) if bat < 60.0 else 0.0
        p_stress = max(0.0, min(1.0, (200.0 - prox) / 150.0)) if prox < 200.0 else 0.0
        t_stress = max(0.0, min(1.0, (temp - 30.0) / 15.0)) if temp > 30.0 else 0.0

        dominant_stress = max(w_stress, b_stress, p_stress)
        overall_stress = dominant_stress * 0.7 + (w_stress + b_stress + p_stress + t_stress) / 4.0 * 0.3

        if prediction == 1 or wind > 8.0 or bat < 20.0 or prox < 50.0:
            calibrated_risk = 0.82 + 0.17 * dominant_stress
            prediction = 1
        else:
            calibrated_risk = 0.015 + 0.635 * (overall_stress ** 1.3)

        return {
            "is_unstable": prediction,
            "risk_probability": round(float(calibrated_risk), 4)
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

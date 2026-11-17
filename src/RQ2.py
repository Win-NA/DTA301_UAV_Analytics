
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.common_ml import save_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class RQ2Service:

    FEATURES = [
        "ambient_temp_C",
        "weather_condition",
        "thermal_signature_intensity",
        "wind_speed_mps"
    ]

    TARGET = "detection_confidence_avg"

    def __init__(self, df, model_path="models/rq2_regressor.pkl"):

        self.df = df.copy()
        self.model_path = model_path

        self.df["weather_condition"] = (
            self.df["weather_condition"]
            .astype("category")
            .cat.codes
        )

        self.X = self.df[self.FEATURES]
        self.y = self.df[self.TARGET]

        self.model = None
        self.metrics = {}

        actual_model_path = Path(self.model_path)
        if not actual_model_path.is_absolute():
            actual_model_path = Path(__file__).resolve().parent.parent / actual_model_path

        if actual_model_path.exists():
            print(f"Loading RQ2 model from: {actual_model_path}")
            saved = joblib.load(actual_model_path)
            if isinstance(saved, dict) and "model" in saved and saved.get("metrics"):
                self.model = saved["model"]
                self.metrics = saved.get("metrics", {})
                print("RQ2 model and training metrics loaded successfully from cache.")
            else:
                print("RQ2 model cache missing metrics. Training on dataset...")
                self.train()
        else:
            print("RQ2 model cache not found. Training on dataset...")
            self.train()

    def train(self):

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42
        )

        baseline = Ridge()
        baseline.fit(X_train, y_train)

        self.model = RandomForestRegressor(
            n_estimators=60,
            max_depth=16,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)

        self.metrics = {
            "r2_score": float(r2_score(y_test, predictions)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions)))
        }

        print("\n========== RQ2 ==========")
        print(self.metrics)

        actual_model_path = Path(self.model_path)
        if not actual_model_path.is_absolute():
            actual_model_path = Path(__file__).resolve().parent.parent / actual_model_path
        joblib.dump({
            "model": self.model,
            "features": self.FEATURES,
            "metrics": self.metrics
        }, actual_model_path, compress=3)

    def predict(self, current):

        weather_map = {"Clear": 0, "Cloudy": 1, "Fog": 2, "Rain": 3}
        raw_weather = current.get("weather_condition", "Clear")
        if isinstance(raw_weather, str):
            weather_code = weather_map.get(raw_weather, 0)
        else:
            weather_code = int(raw_weather)

        input_data = pd.DataFrame([{
            "ambient_temp_C": float(current.get("ambient_temp_C", 25)),
            "weather_condition": weather_code,
            "thermal_signature_intensity": float(current.get(
                "thermal_signature_intensity", 100
            )),
            "wind_speed_mps": float(current.get("wind_speed_mps", 0))
        }])

        prediction = float(self.model.predict(input_data)[0])

        return {
            "predicted_confidence": float(
                np.clip(prediction, 0, 1)
            ),
            "actual_confidence": float(current.get("detection_confidence_avg", prediction))
        }

    def get_metrics(self):
        return self.metrics

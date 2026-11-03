
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
            n_estimators=100,
            random_state=42
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

        save_model(self.model, "rq2_regressor.pkl")

    def predict(self, current):

        input_data = pd.DataFrame([{
            "ambient_temp_C": current.get("ambient_temp_C", 25),
            "weather_condition": current.get("weather_condition_code", 0),
            "thermal_signature_intensity": current.get(
                "thermal_signature_intensity", 0
            ),
            "wind_speed_mps": current.get("wind_speed_mps", 0)
        }])

        prediction = float(self.model.predict(input_data)[0])

        return {
            "predicted_confidence": float(
                np.clip(prediction, 0, 1)
            )
        }

    def get_metrics(self):
        return self.metrics

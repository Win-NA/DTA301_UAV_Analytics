"""
RQ3 Service: Battery Degradation & Energy Anomaly Detection
Course: DTA301 - Data Analytics (Research-Based Learning)
Group: Group 5 (Nhóm 5)

Scientific Principle:
Assesses in-flight battery discharge rate (In-Flight Discharge Rate):
    Discharge Rate (% / min) = (100% - battery_level_pct) / (flight_time_s / 60)
Anomalies (unstable_battery = 1) represent flights where battery capacity is depleted
at an abnormally rapid pace exceeding the 95th percentile operational envelope (p95 threshold).
"""

import os
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


class RQ3Service:

    TARGET = "unstable_battery"

    FEATURES = [
        "battery_level_pct",
        "power_consumption_watts",
        "flight_time_s",
        "wind_speed_mps",
        "ambient_temp_C",
        "altitude_m",
        "weather_code",
    ]

    def __init__(
        self,
        df,
        model_path="models/rq3_energy_model.pkl",
        anomaly_quantile=0.95,
        sample_interval_minutes=15
    ):

        self.raw_df = df.copy()
        self.df = None

        self.model_path = model_path
        self.anomaly_quantile = anomaly_quantile
        self.sample_interval_minutes = sample_interval_minutes

        self.model = None
        self.metrics = {}

        self.available = False
        self.error = None

        self.target_source = "discharge_rate_p95"
        self.threshold = 15.58
        self.threshold_measure = "consumption_pct_per_min"
        self.features = self.FEATURES
        self.labels = pd.Series(dtype=int)

        try:
            self.prepare_data()

            actual_model_path = Path(self.model_path)
            if not actual_model_path.is_absolute():
                actual_model_path = Path(__file__).resolve().parent.parent / actual_model_path

            if actual_model_path.exists():
                print(f"Loading RQ3 model from: {actual_model_path}")
                saved = joblib.load(actual_model_path)
                if isinstance(saved, dict) and "model" in saved:
                    self.model = saved["model"]
                    self.features = saved.get("features", self.FEATURES)
                    self.metrics = saved.get("metrics", {})
                    self.target_source = saved.get("target_source", "discharge_rate_p95")
                    self.threshold = float(saved.get("threshold", 15.58))
                else:
                    self.model = saved
                    self.metrics = {}

                y_full = (self.df[self.threshold_measure] > self.threshold).astype(int)
                self.labels = pd.Series(
                    y_full.values,
                    index=self.df["_source_index"].values
                )
                print("RQ3 model loaded successfully from cache.")
            else:
                self.train()

            self.available = True

        except Exception as exc:
            self.error = str(exc)
            print("RQ3 unavailable:", self.error)

    # =====================================================
    # 1. PREPARE DATA
    # =====================================================

    def prepare_data(self):
        self.df = self.raw_df.copy().reset_index(drop=True)
        self.df["_source_index"] = np.arange(len(self.df))

        # Convert numeric columns safely
        numeric_cols = [
            "battery_level_pct",
            "power_consumption_watts",
            "flight_time_s",
            "wind_speed_mps",
            "ambient_temp_C",
            "altitude_m",
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

        # Weather condition categorical encoding
        weather_map = {"Clear": 0, "Cloudy": 1, "Fog": 2, "Rain": 3}
        self.df["weather_code"] = (
            self.df["weather_condition"].map(weather_map).fillna(0).astype(int)
        )

        # Physical In-Flight Discharge Rate (% pin tiêu hao trên mỗi phút bay)
        flight_min = (self.df["flight_time_s"] / 60.0).clip(lower=1.0)
        self.df["battery_consumed_pct"] = (100.0 - self.df["battery_level_pct"]).clip(lower=0)
        self.df["battery_drop_pct"] = self.df["battery_consumed_pct"]
        self.df["consumption_pct_per_min"] = (self.df["battery_consumed_pct"] / flight_min).fillna(0)

        self.threshold_measure = "consumption_pct_per_min"
        self.target_source = "discharge_rate_p95"
        self.features = self.FEATURES
        self.work = self.df.dropna(subset=[self.threshold_measure]).copy()

    # =====================================================
    # 2. MODEL TRAINING
    # =====================================================

    def train(self):
        actual_model_path = Path(self.model_path)
        if not actual_model_path.is_absolute():
            actual_model_path = Path(__file__).resolve().parent.parent / actual_model_path

        actual_model_path.parent.mkdir(parents=True, exist_ok=True)

        X = self.work[self.features]
        y_measure = self.work[self.threshold_measure]
        self.threshold = float(y_measure.quantile(self.anomaly_quantile))
        y = (y_measure > self.threshold).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)
        proba = rf.predict_proba(X_test)[:, 1]

        self.metrics = {
            "available": True,
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds)),
            "recall": float(recall_score(y_test, preds)),
            "f1_score": float(f1_score(y_test, preds)),
            "auc_roc": float(roc_auc_score(y_test, proba)),
            "train_records": int(len(X_train)),
            "test_records": int(len(X_test)),
            "threshold": round(self.threshold, 2),
            "threshold_measure": self.threshold_measure,
            "target_source": self.target_source,
        }

        self.model = rf
        self.labels = pd.Series(
            (self.df[self.threshold_measure] > self.threshold).astype(int).values,
            index=self.df["_source_index"].values
        )

        joblib.dump(
            {
                "model": self.model,
                "features": self.features,
                "metrics": self.metrics,
                "target_source": self.target_source,
                "threshold": self.threshold
            },
            actual_model_path,
            compress=3
        )
        print("RQ3 in-flight discharge model trained and saved successfully.")

    # =====================================================
    # 3. PREDICTION BY DATASET INDEX
    # =====================================================

    def predict_index(self, index):
        if not self.available:
            return {"available": False, "error": self.error}

        matches = self.df[self.df["_source_index"] == int(index)]
        if matches.empty:
            return {"available": False, "error": "Sample not found."}

        row = matches.iloc[0]

        X_sample = pd.DataFrame([{
            feature: row.get(feature, 0)
            for feature in self.features
        }])

        start = time.perf_counter()
        pred = int(self.model.predict(X_sample)[0])
        proba = float(self.model.predict_proba(X_sample)[0][1])
        latency = (time.perf_counter() - start) * 1000

        rate = float(row.get("consumption_pct_per_min", 0.0))
        bat = float(row.get("battery_level_pct", 100.0))
        ft = float(row.get("flight_time_s", 0.0))
        power = float(row.get("power_consumption_watts", 150.0))
        wind = float(row.get("wind_speed_mps", 0.0))
        temp = float(row.get("ambient_temp_C", 25.0))
        alt = float(row.get("altitude_m", 50.0))

        calibrated_prob = self._calibrate_risk(rate, power, wind, temp, alt, pred, proba)

        return {
            "available": True,
            "unstable_battery": pred,
            "status": "ABNORMAL" if pred == 1 else "NORMAL",
            "abnormal_probability": calibrated_prob,
            "battery_level_pct": bat,
            "battery_drop_pct": round(100.0 - bat, 1),
            "consumption_pct_per_min": round(rate, 2),
            "flight_time_s": ft,
            "power_consumption_watts": power,
            "wind_speed_mps": wind,
            "ambient_temp_C": temp,
            "target_source": self.target_source,
            "threshold": round(self.threshold, 2),
            "energy_anomaly_warning": pred == 1,
            "latency_ms": round(latency, 2)
        }

    def _calibrate_risk(self, rate, power, wind, temp, alt, raw_pred, raw_prob):
        # Discharge rate stress against p95 threshold (15.58 %/min)
        rate_stress = max(0.0, min(1.0, rate / self.threshold))

        # Power stress (nominal 150W, high > 170W, extreme > 200W)
        power_stress = max(0.0, min(1.0, (power - 140.0) / 60.0)) if power > 140.0 else 0.0

        # Wind aerodynamic drag stress (wind > 3 m/s)
        wind_stress = max(0.0, min(1.0, (wind - 3.0) / 7.0)) if wind > 3.0 else 0.0

        # Thermal cell stress (ambient temp > 30°C)
        temp_stress = max(0.0, min(1.0, (temp - 30.0) / 15.0)) if temp > 30.0 else 0.0

        # Altitude air density stress (altitude > 50m)
        alt_stress = max(0.0, min(1.0, (alt - 50.0) / 50.0)) if alt > 50.0 else 0.0

        # Environmental & operational load factor
        load_stress = 0.5 * power_stress + 0.3 * wind_stress + 0.1 * temp_stress + 0.1 * alt_stress

        if raw_pred == 1 or rate >= self.threshold:
            final_risk = 0.82 + 0.17 * min(1.0, 0.6 * (rate / self.threshold) + 0.4 * load_stress)
        else:
            combined_stress = 0.7 * (rate_stress ** 1.4) + 0.3 * load_stress
            final_risk = 0.008 + 0.642 * (combined_stress ** 1.3)

        return round(float(final_risk), 4)

    # =====================================================
    # 4. PREDICT DICTIONARY
    # =====================================================

    def predict(self, current):
        if not self.available:
            return {"available": False, "error": self.error}

        weather_map = {"Clear": 0, "Cloudy": 1, "Fog": 2, "Rain": 3}
        current_dict = dict(current)
        if "weather_code" not in current_dict:
            raw_w = current_dict.get("weather_condition", "Clear")
            current_dict["weather_code"] = weather_map.get(raw_w, 0)

        X_sample = pd.DataFrame([{
            feature: current_dict.get(feature, 0)
            for feature in self.features
        }])

        pred = int(self.model.predict(X_sample)[0])
        proba = float(self.model.predict_proba(X_sample)[0][1])

        bat = float(current_dict.get("battery_level_pct", 100.0))
        ft = float(current_dict.get("flight_time_s", 0.0))
        power = float(current_dict.get("power_consumption_watts", 150.0))
        wind = float(current_dict.get("wind_speed_mps", 0.0))
        temp = float(current_dict.get("ambient_temp_C", 25.0))
        alt = float(current_dict.get("altitude_m", 50.0))
        ft_min = max(1.0, ft / 60.0)
        rate = (100.0 - bat) / ft_min

        calibrated_prob = self._calibrate_risk(rate, power, wind, temp, alt, pred, proba)

        return {
            "available": True,
            "unstable_battery": pred,
            "status": "ABNORMAL" if pred == 1 else "NORMAL",
            "abnormal_probability": calibrated_prob,
            "consumption_pct_per_min": round(rate, 2),
            "threshold": round(self.threshold, 2)
        }

    # =====================================================
    # 5. HISTORY FOR DASHBOARD
    # =====================================================

    def get_history(self, index, window=30):
        matches = self.df[self.df["_source_index"] == int(index)]
        if matches.empty:
            return []

        current = matches.iloc[0]
        drone_history = self.df[self.df["drone_id"] == current["drone_id"]].copy()
        if "timestamp" in drone_history.columns:
            drone_history = drone_history.sort_values("timestamp")
            history = drone_history[drone_history["timestamp"] <= current["timestamp"]].tail(window)
        else:
            history = drone_history[drone_history["_source_index"] <= index].tail(window)

        columns = [
            "timestamp",
            "mission_id",
            "battery_level_pct",
            "battery_drop_pct",
            "consumption_pct_per_min",
            "power_consumption_watts",
            "wind_speed_mps",
            "ambient_temp_C"
        ]
        columns = [c for c in columns if c in history.columns]
        res = history[columns].copy()
        res["timestamp"] = res["timestamp"].astype(str)
        return res.astype(object).where(pd.notna(res), None).to_dict("records")

    # =====================================================
    # 6. SUMMARY & METRICS
    # =====================================================

    def get_consumption_summary(self):
        return {
            "available": True,
            "records": int(len(self.df)),
            "mean_consumption_pct_per_min": float(self.df["consumption_pct_per_min"].mean()),
            "median_consumption_pct_per_min": float(self.df["consumption_pct_per_min"].median()),
            "max_consumption_pct_per_min": float(self.df["consumption_pct_per_min"].max()),
            "mean_power_consumption_watts": float(self.df["power_consumption_watts"].mean()),
        }

    def get_metrics(self):
        if not self.available:
            return {"available": False, "error": self.error}
        return self.metrics

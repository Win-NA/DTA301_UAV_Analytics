
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class RQ3Service:

    TARGET = "unstable_battery"

    BATTERY_ALIASES = [
        "battery_level_pct",
        "battery_level",
        "battery_level_percent"
    ]

    NUMERIC_CANDIDATES = [
        "battery_level_pct_prev",
        "battery_drop_lag1",
        "battery_drop_lag2",
        "power_consumption_watts_prev",
        "wind_speed_mps_prev",
        "ambient_temp_C_prev",
        "altitude_m_prev",
        "distance_to_base_m_prev",
        "hover_duration_s_prev",
        "flight_time_s_prev",
        "camera_active_prev",
        "velocity_magnitude_prev"
    ]

    CATEGORICAL_CANDIDATES = [
        "weather_condition_prev",
        "mission_type_prev",
        "recording_mode_prev",
        "surveillance_pattern_prev",
        "region_type_prev"
    ]

    CURRENT_NUMERIC_FEATURES = [
        "battery_level_pct",
        "wind_speed_mps",
        "ambient_temp_C",
        "altitude_m",
        "distance_to_base_m",
        "hover_duration_s",
        "flight_time_s",
        "camera_active",
        "velocity_magnitude"
    ]

    CURRENT_CATEGORICAL_FEATURES = [
        "weather_condition",
        "mission_type",
        "recording_mode",
        "surveillance_pattern",
        "region_type"
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

        self.target_source = None
        self.threshold = None
        self.threshold_measure = None

        self.features = []
        self.numeric_features = []
        self.categorical_features = []

        self.group_columns = []
        self.sequence_available = False
        self.labels = pd.Series(dtype=int)

        try:
            self.prepare_data()
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

        self.df["_source_index"] = np.arange(
            len(self.df)
        )

        # ---------------------------------------------
        # Battery column mapping
        # ---------------------------------------------

        battery_col = next(
            (
                col for col in self.BATTERY_ALIASES
                if col in self.df.columns
            ),
            None
        )

        if battery_col is None:

            raise ValueError(
                "Battery column not found. Expected "
                "battery_level_pct, battery_level, or "
                "battery_level_percent."
            )

        self.df["battery_level_pct"] = pd.to_numeric(
            self.df[battery_col],
            errors="coerce"
        )

        # ---------------------------------------------
        # Timestamp conversion
        # ---------------------------------------------

        if "timestamp" in self.df.columns:

            self.df["timestamp"] = pd.to_datetime(
                self.df["timestamp"],
                errors="coerce"
            )

        # ---------------------------------------------
        # Numeric conversion
        # ---------------------------------------------

        numeric_columns = [
            "battery_level_pct",
            "power_consumption_watts",
            "wind_speed_mps",
            "ambient_temp_C",
            "altitude_m",
            "distance_to_base_m",
            "hover_duration_s",
            "flight_time_s",
            "camera_active",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "acceleration_x",
            "acceleration_y",
            "acceleration_z"
        ]

        for col in numeric_columns:

            if col in self.df.columns:

                self.df[col] = pd.to_numeric(
                    self.df[col],
                    errors="coerce"
                )

        # ---------------------------------------------
        # Velocity magnitude
        # ---------------------------------------------

        velocity_cols = [
            "velocity_x",
            "velocity_y",
            "velocity_z"
        ]

        if all(
            col in self.df.columns
            for col in velocity_cols
        ):

            self.df["velocity_magnitude"] = np.sqrt(
                sum(
                    self.df[col] ** 2
                    for col in velocity_cols
                )
            )

        # ---------------------------------------------
        # Select safe grouping strategy
        # ---------------------------------------------

        group_candidates = []

        if (
            "drone_id" in self.df.columns
            and "mission_id" in self.df.columns
        ):

            group_candidates.append([
                "drone_id",
                "mission_id"
            ])

        if "mission_id" in self.df.columns:

            group_candidates.append([
                "mission_id"
            ])

        if "drone_id" in self.df.columns:

            # Only use drone-only grouping when no mission ID
            # exists; otherwise, do not cross mission boundaries.
            if "mission_id" not in self.df.columns:
                group_candidates.append(["drone_id"])

        self.group_columns = []

        for candidate in group_candidates:

            counts = self.df.groupby(
                candidate,
                dropna=False
            ).size()

            transitions = int(
                (counts - 1).clip(lower=0).sum()
            )

            if transitions >= 30:

                self.group_columns = candidate
                break

        # ---------------------------------------------
        # Sort chronologically
        # ---------------------------------------------

        sort_columns = self.group_columns.copy()

        if "timestamp" in self.df.columns:
            sort_columns.append("timestamp")

        if sort_columns:

            self.df = self.df.sort_values(
                sort_columns,
                kind="stable",
                na_position="last"
            ).reset_index(drop=True)

        # ---------------------------------------------
        # Battery consumption
        # ---------------------------------------------

        if self.group_columns:

            self.sequence_available = True

            grouped = self.df.groupby(
                self.group_columns,
                dropna=False,
                sort=False
            )

            previous_battery = (
                grouped["battery_level_pct"].shift(1)
            )

            self.df["battery_level_pct_prev"] = (
                previous_battery
            )

            self.df["battery_drop_pct"] = (
                previous_battery -
                self.df["battery_level_pct"]
            )

            self.df["battery_consumption"] = (
                self.df["battery_drop_pct"].clip(lower=0)
            )

            self.df["battery_drop_lag1"] = (
                self.df.groupby(
                    self.group_columns,
                    dropna=False
                )["battery_consumption"].shift(1)
            )

            self.df["battery_drop_lag2"] = (
                self.df.groupby(
                    self.group_columns,
                    dropna=False
                )["battery_consumption"].shift(2)
            )

            if "timestamp" in self.df.columns:

                previous_time = (
                    self.df.groupby(
                        self.group_columns,
                        dropna=False
                    )["timestamp"].shift(1)
                )

                interval = (
                    self.df["timestamp"] - previous_time
                ).dt.total_seconds() / 60

                self.df["interval_minutes"] = interval.where(
                    interval > 0,
                    np.nan
                )

            else:

                self.df["interval_minutes"] = (
                    self.sample_interval_minutes
                )

            self.df["consumption_pct_per_min"] = (
                self.df["battery_consumption"] /
                self.df["interval_minutes"]
            )

            # Previous telemetry features
            previous_columns = [
                "power_consumption_watts",
                "wind_speed_mps",
                "ambient_temp_C",
                "altitude_m",
                "distance_to_base_m",
                "hover_duration_s",
                "flight_time_s",
                "camera_active",
                "velocity_magnitude",
                "weather_condition",
                "mission_type",
                "recording_mode",
                "surveillance_pattern",
                "region_type"
            ]

            for col in previous_columns:

                if col in self.df.columns:

                    self.df[f"{col}_prev"] = (
                        self.df.groupby(
                            self.group_columns,
                            dropna=False
                        )[col].shift(1)
                    )

            valid_rates = (
                self.df["consumption_pct_per_min"]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if len(valid_rates) < 30:
                self.sequence_available = False

        else:

            self.sequence_available = False

            self.df["battery_drop_pct"] = np.nan

            self.df["battery_consumption"] = np.nan

            self.df["consumption_pct_per_min"] = np.nan

        # =================================================
        # 2. TARGET
        # =================================================

        if self.TARGET in self.df.columns:

            raw_target = self.df[self.TARGET]

            if pd.api.types.is_numeric_dtype(raw_target):

                self.df[self.TARGET] = pd.to_numeric(
                    raw_target,
                    errors="coerce"
                )

            else:

                label_map = {
                    "normal": 0,
                    "stable": 0,
                    "0": 0,
                    "abnormal": 1,
                    "unstable": 1,
                    "anomaly": 1,
                    "1": 1
                }

                self.df[self.TARGET] = (
                    raw_target.astype(str)
                    .str.strip()
                    .str.lower()
                    .map(label_map)
                )

            self.target_source = "dataset_label"

            self.threshold_measure = None

        else:

            if self.sequence_available:

                self.threshold_measure = (
                    "consumption_pct_per_min"
                )

                self.target_source = "heuristic_battery_rate_p95"

            elif "power_consumption_watts" in self.df.columns:

                self.threshold_measure = (
                    "power_consumption_watts"
                )

                self.target_source = "heuristic_power_p95"

            else:

                raise ValueError(
                    "Cannot generate RQ3 target. "
                    "The dataset needs repeated battery samples "
                    "or power_consumption_watts."
                )

        # =================================================
        # 3. SELECT FEATURES
        # =================================================

        if self.sequence_available:

            numeric_candidates = self.NUMERIC_CANDIDATES

            categorical_candidates = (
                self.CATEGORICAL_CANDIDATES
            )

        else:

            # No reliable repeated battery sequence:
            # use current flight telemetry to assess measured
            # power consumption instead.
            numeric_candidates = (
                self.CURRENT_NUMERIC_FEATURES
            )

            categorical_candidates = (
                self.CURRENT_CATEGORICAL_FEATURES
            )

        self.numeric_features = [
            col for col in numeric_candidates
            if (
                col in self.df.columns
                and self.df[col].notna().any()
            )
        ]

        self.categorical_features = [
            col for col in categorical_candidates
            if (
                col in self.df.columns
                and self.df[col].notna().any()
            )
        ]

        self.features = (
            self.numeric_features +
            self.categorical_features
        )

        if not self.features:

            raise ValueError(
                "No usable RQ3 input features were found."
            )

        # Numeric cleanup
        for col in self.numeric_features:

            self.df[col] = pd.to_numeric(
                self.df[col],
                errors="coerce"
            )

        # Keep target rows
        if self.target_source == "dataset_label":

            self.work = self.df.dropna(
                subset=[self.TARGET]
            ).copy()

        else:

            self.work = self.df.dropna(
                subset=[self.threshold_measure]
            ).copy()

        if len(self.work) < 30:

            raise ValueError(
                f"Only {len(self.work)} valid RQ3 rows. "
                "At least 30 rows are required."
            )

        self.work = self.work.sort_values(
            "_source_index",
            kind="stable"
        ).reset_index(drop=True)

    # =====================================================
    # 4. MODEL TRAINING
    # =====================================================

    def train(self):

        os.makedirs(
            os.path.dirname(self.model_path) or ".",
            exist_ok=True
        )

        split = int(len(self.work) * 0.8)

        train_df = self.work.iloc[:split].copy()
        test_df = self.work.iloc[split:].copy()

        if len(train_df) < 10 or len(test_df) < 1:

            raise ValueError(
                "Not enough records for chronological evaluation."
            )

        # Generate labels.
        if self.target_source == "dataset_label":

            y_train = train_df[self.TARGET].astype(int)

            y_test = test_df[self.TARGET].astype(int)

        else:

            train_measure = train_df[
                self.threshold_measure
            ]

            self.threshold = float(
                train_measure.quantile(
                    self.anomaly_quantile
                )
            )

            if not np.isfinite(self.threshold):

                raise ValueError(
                    "Unable to calculate RQ3 anomaly threshold."
                )

            y_train = (
                train_measure > self.threshold
            ).astype(int)

            y_test = (
                test_df[self.threshold_measure]
                > self.threshold
            ).astype(int)

        if y_train.nunique() < 2:

            raise ValueError(
                "RQ3 training period contains only one target class. "
                "Review the data sequence and anomaly threshold."
            )

        X_train = train_df[self.features]
        X_test = test_df[self.features]

        # -------------------------------------------------
        # Preprocessing
        # -------------------------------------------------

        transformers = []

        if self.numeric_features:

            transformers.append((
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    )
                ]),
                self.numeric_features
            ))

        if self.categorical_features:

            transformers.append((
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore")
                    )
                ]),
                self.categorical_features
            ))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        # -------------------------------------------------
        # Random Forest
        # -------------------------------------------------

        self.model = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ])

        self.model.fit(
            X_train,
            y_train
        )

        predictions = self.model.predict(
            X_test
        )

        self.metrics = {

            "available": True,

            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),

            "precision": float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                )
            ),

            "recall": float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                )
            ),

            "f1_score": float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                )
            ),

            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1]
            ).tolist(),

            "train_records": int(len(X_train)),

            "test_records": int(len(X_test)),

            "target_source": self.target_source,

            "threshold": self.threshold,

            "threshold_measure": self.threshold_measure,

            "split_method": "chronological_holdout",

            "features": self.features,

            "sequence_available": self.sequence_available,

            "group_columns": self.group_columns

        }

        # -------------------------------------------------
        # Feature importance
        # -------------------------------------------------

        try:

            sample_size = min(1500, len(X_test))

            sample_X = X_test.sample(
                n=sample_size,
                random_state=42
            )

            sample_y = y_test.loc[
                sample_X.index
            ]

            importance = permutation_importance(
                self.model,
                sample_X,
                sample_y,
                n_repeats=3,
                random_state=42,
                scoring="f1",
                n_jobs=-1
            )

            self.metrics["feature_importance"] = {

                feature: float(value)

                for feature, value in sorted(
                    zip(
                        self.features,
                        importance.importances_mean
                    ),
                    key=lambda item: item[1],
                    reverse=True
                )

            }

        except Exception as exc:

            print(
                "RQ3 feature importance unavailable:",
                exc
            )

            self.metrics["feature_importance"] = {}

        # -------------------------------------------------
        # Refit full dataset for dashboard predictions
        # -------------------------------------------------

        X_full = self.work[self.features]

        if self.target_source == "dataset_label":

            y_full = self.work[self.TARGET].astype(int)

        else:

            y_full = (
                self.work[self.threshold_measure]
                > self.threshold
            ).astype(int)

        self.labels = pd.Series(
            y_full.values,
            index=self.work["_source_index"].values
        )

        self.model.fit(
            X_full,
            y_full
        )

        joblib.dump(
            {
                "model": self.model,
                "features": self.features,
                "metrics": self.metrics,
                "target_source": self.target_source,
                "threshold": self.threshold
            },
            self.model_path
        )

        print("RQ3 model trained successfully.")
        print("RQ3 target:", self.target_source)
        print("RQ3 features:", self.features)

    # =====================================================
    # 5. PREDICTION BY DATASET INDEX
    # =====================================================

    def predict_index(self, index):

        if not self.available:

            return {
                "available": False,
                "error": self.error
            }

        matches = self.df[
            self.df["_source_index"] == int(index)
        ]

        if matches.empty:

            return {
                "available": False,
                "error": "Sample not found."
            }

        row = matches.iloc[0]

        X = pd.DataFrame([
            {
                feature: row.get(feature, np.nan)
                for feature in self.features
            }
        ])

        start = time.perf_counter()

        prediction = int(
            self.model.predict(X)[0]
        )

        probabilities = self.model.predict_proba(X)[0]

        classes = list(
            self.model.classes_
        )

        positive_index = (
            classes.index(1)
            if 1 in classes else None
        )

        abnormal_probability = (
            float(probabilities[positive_index])
            if positive_index is not None
            else 0.0
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        return {

            "available": True,

            "unstable_battery": prediction,

            "status": (
                "ABNORMAL"
                if prediction == 1
                else "NORMAL"
            ),

            "abnormal_probability": abnormal_probability,

            "battery_level_pct": self._number(
                row.get("battery_level_pct")
            ),

            "battery_drop_pct": self._number(
                row.get("battery_drop_pct")
            ),

            "consumption_pct_per_min": self._number(
                row.get("consumption_pct_per_min")
            ),

            "power_consumption_watts": self._number(
                row.get("power_consumption_watts")
            ),

            "wind_speed_mps": self._number(
                row.get("wind_speed_mps")
            ),

            "ambient_temp_C": self._number(
                row.get("ambient_temp_C")
            ),

            "target_source": self.target_source,

            "threshold": self.threshold,

            "sequence_available": self.sequence_available,

            "energy_anomaly_warning": prediction == 1,

            "rth_warning": None,

            "latency_ms": round(latency, 2)

        }

    # =====================================================
    # 6. LEGACY PREDICT METHOD
    # =====================================================

    def predict(self, current):

        if not self.available:

            return {
                "available": False,
                "error": self.error
            }

        X = pd.DataFrame([
            {
                feature: current.get(
                    feature,
                    np.nan
                )
                for feature in self.features
            }
        ])

        start = time.perf_counter()

        prediction = int(
            self.model.predict(X)[0]
        )

        probability = self.model.predict_proba(X)[0]

        classes = list(self.model.classes_)

        positive_index = (
            classes.index(1)
            if 1 in classes else None
        )

        return {

            "available": True,

            "unstable_battery": prediction,

            "status": (
                "ABNORMAL"
                if prediction == 1
                else "NORMAL"
            ),

            "abnormal_probability": (
                float(probability[positive_index])
                if positive_index is not None
                else 0.0
            ),

            "target_source": self.target_source,

            "energy_anomaly_warning": prediction == 1,

            "rth_warning": None,

            "latency_ms": round(
                (time.perf_counter() - start) * 1000,
                2
            )

        }

    # =====================================================
    # 7. HISTORY FOR DASHBOARD
    # =====================================================

    def get_history(self, index, window=30):

        matches = self.df[
            self.df["_source_index"] == int(index)
        ]

        if matches.empty:
            return []

        current = matches.iloc[0]

        history = self.df.copy()

        # Same mission/drone only.
        for col in self.group_columns:

            value = current[col]

            if pd.isna(value):

                history = history[
                    history[col].isna()
                ]

            else:

                history = history[
                    history[col] == value
                ]

        history = history[
            history["_source_index"] <= index
        ].tail(window)

        columns = [
            "timestamp",
            "battery_level_pct",
            "battery_drop_pct",
            "consumption_pct_per_min",
            "power_consumption_watts",
            "wind_speed_mps",
            "ambient_temp_C"
        ]

        columns = [
            col for col in columns
            if col in history.columns
        ]

        result = history[columns].copy()

        if "timestamp" in result.columns:

            result["timestamp"] = (
                result["timestamp"].astype(str)
            )

        result = result.astype(object).where(
            pd.notna(result),
            None
        )

        return result.to_dict(
            orient="records"
        )

    # =====================================================
    # 8. SUMMARY
    # =====================================================

    def get_consumption_summary(self):

        if not self.available:

            return {
                "available": False,
                "error": self.error
            }

        rates = pd.to_numeric(
            self.df["consumption_pct_per_min"],
            errors="coerce"
        ).replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        power = pd.to_numeric(
            self.df.get(
                "power_consumption_watts",
                pd.Series(dtype=float)
            ),
            errors="coerce"
        ).dropna()

        return {

            "available": True,

            "records": int(len(self.df)),

            "mean_consumption_pct_per_min": (
                float(rates.mean())
                if len(rates) else None
            ),

            "median_consumption_pct_per_min": (
                float(rates.median())
                if len(rates) else None
            ),

            "max_consumption_pct_per_min": (
                float(rates.max())
                if len(rates) else None
            ),

            "mean_power_consumption_watts": (
                float(power.mean())
                if len(power) else None
            ),

            "abnormal_rate": (
                float(self.labels.mean())
                if len(self.labels) else None
            ),

            "target_source": self.target_source,

            "sequence_available": self.sequence_available

        }

    # =====================================================
    # 9. METRICS
    # =====================================================

    def get_metrics(self):

        if not self.available:

            return {
                "available": False,
                "error": self.error
            }

        return self.metrics

    # =====================================================
    # 10. SAFE NUMERIC CONVERSION
    # =====================================================

    @staticmethod
    def _number(value):

        try:

            number = float(value)

            if np.isfinite(number):
                return number

        except (TypeError, ValueError):

            pass

        return None

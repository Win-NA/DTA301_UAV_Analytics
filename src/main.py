
import os
import time
import random
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.RQ1 import RQ1Service
from src.RQ2 import RQ2Service
from src.RQ3 import RQ3Service


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="DTA301 Drone Surveillance API",
    description="Three Research Question ML API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# DATASET
# ============================================================

possible_data_paths = [
    "data/processed/SurveilDrone_Net23_cleaned.csv",
    "../data/processed/SurveilDrone_Net23_cleaned.csv",
    "data/SurveilDrone-Net23.csv",
    "../data/raw/SurveilDrone-Net23.csv"
]

data_path = next(
    (path for path in possible_data_paths if os.path.exists(path)),
    None
)

if data_path is None:
    raise FileNotFoundError(
        "Dataset not found. Check data/processed/"
    )

df = pd.read_csv(data_path)

print("Dataset loaded:", data_path)
print("Dataset shape:", df.shape)


# ============================================================
# SERVICES
# ============================================================

rq1_service = RQ1Service(df)
rq2_service = RQ2Service(df)
rq3_service = RQ3Service(df)


# ============================================================
# DRONE IMAGES
# ============================================================

DRONE_IMAGES = [
    {
        "url": "https://images.unsplash.com/photo-1508614589041-895b88991e3e?w=800&q=80",
        "desc": "Drone surveying solar energy area"
    },
    {
        "url": "https://images.unsplash.com/photo-1509391365360-2e959784a276?w=800&q=80",
        "desc": "Thermal camera detecting hotspots"
    },
    {
        "url": "https://images.unsplash.com/photo-1613665813446-82a78c468a1d?w=800&q=80",
        "desc": "Solar panel inspection"
    },
    {
        "url": "https://images.unsplash.com/photo-1548337138-e87d889cc369?w=800&q=80",
        "desc": "Solar panel shading inspection"
    }
]


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def read_root():

    base_dir = os.path.dirname(__file__)

    html_path = os.path.join(base_dir, "index.html")

    if os.path.exists(html_path):
        return FileResponse(html_path)

    return {
        "message": "DTA301 API running",
        "docs": "/docs"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/v1/health")
def health():

    return {
        "status": "running",
        "rq1": True,
        "rq2": True,
        "rq3": rq3_service.available
    }


# ============================================================
# ALL METRICS
# ============================================================

@app.get("/api/v1/metrics")
def get_metrics():

    return {
        "rq1": rq1_service.get_metrics(),
        "rq2": rq2_service.get_metrics(),
        "rq3": rq3_service.get_metrics(),
        "rq3_summary": rq3_service.get_consumption_summary() if rq3_service.available else {},
        "dataset_size": len(df)
    }


# ============================================================
# RQ1
# ============================================================

@app.get("/api/v1/rq1/metrics")
def rq1_metrics():

    return rq1_service.get_metrics()


@app.get("/api/v1/rq1/predict/{idx}")
def rq1_predict(idx: int):

    current = df.iloc[idx % len(df)].to_dict()

    return rq1_service.predict(current)


@app.get("/api/v1/rq1/feature-importance")
def rq1_feature_importance():

    return rq1_service.get_feature_importance()


# ============================================================
# RQ2
# ============================================================

@app.get("/api/v1/rq2/metrics")
def rq2_metrics():

    return rq2_service.get_metrics()


@app.get("/api/v1/rq2/predict/{idx}")
def rq2_predict(idx: int):

    current = df.iloc[idx % len(df)].to_dict()

    return rq2_service.predict(current)


# ============================================================
# RQ3
# ============================================================

@app.get("/api/v1/rq3/metrics")
def rq3_metrics():

    if not rq3_service.available:
        raise HTTPException(
            status_code=503,
            detail=getattr(rq3_service, "error", "RQ3 is unavailable")
        )

    return rq3_service.get_metrics()


@app.get("/api/v1/rq3/predict/{idx}")
def rq3_predict(idx: int):

    if not rq3_service.available:
        raise HTTPException(
            status_code=503,
            detail=getattr(rq3_service, "error", "RQ3 is unavailable")
        )

    idx = idx % len(df)

    return rq3_service.predict_index(idx)


@app.get("/api/v1/rq3/history/{idx}")
def rq3_history(idx: int, window: int = 30):

    if not rq3_service.available:
        raise HTTPException(
            status_code=503,
            detail=getattr(rq3_service, "error", "RQ3 is unavailable")
        )

    idx = idx % len(df)
    window = max(1, min(window, 200))

    return rq3_service.get_history(
        idx,
        window=window
    )


@app.get("/api/v1/rq3/summary")
def rq3_summary():

    if not rq3_service.available:
        raise HTTPException(
            status_code=503,
            detail=getattr(rq3_service, "error", "RQ3 is unavailable")
        )

    return rq3_service.get_consumption_summary()

# ============================================================
# TELEMETRY
# ============================================================

@app.get("/api/v1/telemetry/{idx}")
def get_telemetry_sample(idx: int):

    start_time = time.perf_counter()

    idx = idx % len(df)

    current = df.iloc[idx].to_dict()

    # RQ1 prediction
    rq1_prediction = rq1_service.predict(current)

    # RQ2 prediction
    rq2_prediction = rq2_service.predict(current)

    # RQ3 prediction
    # RQ3 prediction
    if rq3_service.available:
        rq3_prediction = rq3_service.predict_index(idx)
        rq3_history_data = rq3_service.get_history(
            idx,
            window=30
        )
        rq3_summary_data = rq3_service.get_consumption_summary()
        rq3_status = "available"
    else:
        rq3_prediction = {
            "available": False,
            "error": getattr(rq3_service, "error", "RQ3 unavailable")
        }
        rq3_history_data = []
        rq3_summary_data = {}
        rq3_status = "unavailable"

    img_info = DRONE_IMAGES[
        idx % len(DRONE_IMAGES)
    ]

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return {
        "sample_index": idx,

        "timestamp": str(current.get("timestamp", "")),

        "mission_id": str(current.get("mission_id", "")),

        "drone_id": str(current.get("drone_id", "")),

        "telemetry": {
            "battery_level_pct": float(
                current.get("battery_level_pct", 0)
            ),

            "wind_speed_mps": float(
                current.get("wind_speed_mps", 0)
            ),

            "ambient_temp_C": float(
                current.get("ambient_temp_C", 0)
            ),

            "weather_condition": str(
                current.get("weather_condition", "Clear")
            ),

            "altitude_m": float(
                current.get("altitude_m", 0)
            ),

            "distance_to_base_m": float(
                current.get("distance_to_base_m", 0)
            ),

            "proximity_to_restricted_zone_m": float(
                current.get("proximity_to_restricted_zone_m", 0)
            ),

            "gps_lat": float(
                current.get("gps_lat", 0)
            ),

            "gps_lon": float(
                current.get("gps_lon", 0)
            ),

            "detected_object_count": int(
                current.get("detected_object_count", 0)
            ),

            "target_type": str(
                current.get("target_type", "Unknown")
            ),

            "thermal_signature_intensity": float(
                current.get("thermal_signature_intensity", 100.0)
            ),

            "detection_confidence_avg": float(
                current.get("detection_confidence_avg", 0.65)
            ),

            "power_consumption_watts": float(
                current.get("power_consumption_watts", 150.0)
            )
        },

        "rq1_prediction": rq1_prediction,

        "rq2_prediction": rq2_prediction,

        "rq3_prediction": rq3_prediction,

        "rq3_history": rq3_history_data,

        "rq3_summary": rq3_summary_data,

        "rq3_metrics": (
            rq3_service.get_metrics()
            if rq3_service.available
            else {}
        ),

        "rq3_status": rq3_status,

        "image": img_info,
    }


# ============================================================
# DATASET INFORMATION & DRONE FLEET
# ============================================================

# Precompute chronological indices for each drone
drone_ids = sorted([str(d) for d in df["drone_id"].dropna().unique().tolist()])
drone_indices = {}
for d in drone_ids:
    sub = df[df["drone_id"] == d]
    if "timestamp" in sub.columns:
        drone_indices[d] = [int(x) for x in sub.sort_values("timestamp").index.tolist()]
    else:
        drone_indices[d] = [int(x) for x in sub.index.tolist()]


@app.get("/api/v1/datasets")
def get_datasets():

    return {
        "dataset_name": os.path.basename(data_path),
        "dataset_size": len(df),
        "columns": list(df.columns),
        "rq1_features": rq1_service.FEATURES,
        "rq2_features": rq2_service.FEATURES,
        "rq3_features": rq3_service.FEATURES
    }


@app.get("/api/v1/drones")
def get_drones():
    """Return all unique drones and their precomputed sample counts."""
    return [
        {
            "drone_id": d,
            "total_samples": len(drone_indices[d]),
            "first_index": drone_indices[d][0] if drone_indices[d] else 0
        }
        for d in drone_ids
    ]


@app.get("/api/v1/drone/{drone_id}/indices")
def get_drone_indices(drone_id: str):
    """Return the full chronological list of row indices for a specific drone."""
    if drone_id not in drone_indices:
        raise HTTPException(
            status_code=404,
            detail=f"Drone '{drone_id}' not found in fleet"
        )
    return {
        "drone_id": drone_id,
        "indices": drone_indices[drone_id],
        "total": len(drone_indices[drone_id])
    }


@app.get("/api/v1/sample/risk")
def get_risk_sample(drone_id: str = None):
    """Find a random unstable/risky flight telemetry sample."""
    target_df = (
        df[df["drone_id"] == drone_id]
        if (drone_id and drone_id != "ALL" and drone_id in drone_indices)
        else df
    )
    risk_rows = target_df[target_df["unstable_flight"] == 1]
    if risk_rows.empty:
        risk_rows = df[df["unstable_flight"] == 1]
    chosen = risk_rows.sample(n=1)
    return {
        "index": int(chosen.index[0]),
        "drone_id": str(chosen["drone_id"].iloc[0])
    }


@app.get("/api/v1/sample/safe")
def get_safe_sample(drone_id: str = None):
    """Find a random stable/safe flight telemetry sample."""
    target_df = (
        df[df["drone_id"] == drone_id]
        if (drone_id and drone_id != "ALL" and drone_id in drone_indices)
        else df
    )
    safe_rows = target_df[target_df["unstable_flight"] == 0]
    if safe_rows.empty:
        safe_rows = df[df["unstable_flight"] == 0]
    chosen = safe_rows.sample(n=1)
    return {
        "index": int(chosen.index[0]),
        "drone_id": str(chosen["drone_id"].iloc[0])
    }


@app.get("/api/v1/sample/battery_anomaly")
def get_battery_anomaly_sample(drone_id: str = None):
    """Find an actual sample where RQ3 predicts energy anomaly spike (unstable_battery == 1)."""
    if rq3_service.available and len(rq3_service.labels) > 0:
        anom_indices = rq3_service.labels[rq3_service.labels == 1].index
        if drone_id and drone_id != "ALL" and drone_id in drone_indices:
            anom_indices = [idx for idx in anom_indices if df.loc[idx, "drone_id"] == drone_id]
        if len(anom_indices) > 0:
            chosen_idx = int(random.choice(list(anom_indices)))
            return {
                "index": chosen_idx,
                "drone_id": str(df.loc[chosen_idx, "drone_id"]),
                "battery_level_pct": float(df.loc[chosen_idx, "battery_level_pct"])
            }

    target_df = (
        df[df["drone_id"] == drone_id]
        if (drone_id and drone_id != "ALL" and drone_id in drone_indices)
        else df
    )
    anomaly_rows = target_df[target_df["battery_level_pct"] < 20]
    if anomaly_rows.empty:
        anomaly_rows = df[df["battery_level_pct"] < 20]
    chosen = anomaly_rows.sample(n=1)
    return {
        "index": int(chosen.index[0]),
        "drone_id": str(chosen["drone_id"].iloc[0]),
        "battery_level_pct": float(chosen["battery_level_pct"].iloc[0])
    }



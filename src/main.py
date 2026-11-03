
import os
import time
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
# DATASET INFORMATION
# ============================================================

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

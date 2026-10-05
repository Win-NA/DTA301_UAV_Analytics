import os
import time
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

app = FastAPI(title="DTA301 Drone Surveillance Single-Dataset API")

# Cấu hình CORS cho phép Frontend giao tiếp API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. PHẦN TẢI VÀ ĐỊNH NGHĨA ROUTE TRANG CHỦ (INDEX.HTML)
# ==========================================
@app.get("/")
def read_root():
    # Tự động định vị file index.html nằm cùng thư mục với main.py (src/)
    base_dir = os.path.dirname(__file__)
    html_path = os.path.join(base_dir, "index.html")
    if not os.path.exists(html_path):
        html_path = os.path.join(base_dir, "..", "index.html")
        
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Server đang chạy. Hãy đảm bảo file index.html nằm trong thư mục src/."}


# ==========================================
# 2. LOAD DATASET SẠCH VÀ SỬ DỤNG MÔ HÌNH DTA301
# ==========================================
possible_data_paths = [
    "data/processed/SurveilDrone_Net23_cleaned.csv",
    "../data/processed/SurveilDrone_Net23_cleaned.csv",
    "data/SurveilDrone-Net23.csv",
    "../data/raw/SurveilDrone-Net23.csv"
]

data_path = None
for p in possible_data_paths:
    if os.path.exists(p):
        data_path = p
        break

if data_path is None:
    raise FileNotFoundError("❌ Không tìm thấy dataset! Hãy kiểm tra lại file CSV trong data/processed/ hoặc data/raw/.")

df = pd.read_csv(data_path)

# Tạo nhãn biến mục tiêu unstable_flight nếu chưa có sẵn
if 'unstable_flight' not in df.columns:
    df['unstable_flight'] = (
        (df['battery_level_pct'] < 20) | 
        (df['wind_speed_mps'] > 8.0) | 
        (df['proximity_to_restricted_zone_m'] < 50)
    ).astype(int)

# Định nghĩa danh sách thuộc tính chuẩn
clf_features = ['battery_level_pct', 'wind_speed_mps', 'proximity_to_restricted_zone_m', 'ambient_temp_C']
reg_features = ['ambient_temp_C', 'wind_speed_mps', 'detected_object_count', 'thermal_signature_intensity']

# Kiểm tra & Tải các mô hình .pkl đã train từ Phase 3
model_dir = "models" if os.path.exists("models") else "../models"
rq1_path = os.path.join(model_dir, "rq1_classifier.pkl")
rq2_path = os.path.join(model_dir, "rq2_regressor.pkl")

# Load hoặc Train Mô hình RQ1
if os.path.exists(rq1_path):
    clf_model = joblib.load(rq1_path)
else:
    X_clf = df[clf_features]
    y_clf = df['unstable_flight']
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_clf, y_clf)

# Tính chỉ số RQ1
y_clf_pred = clf_model.predict(df[clf_features])
rq1_acc = float(accuracy_score(df['unstable_flight'], y_clf_pred))
rq1_f1 = float(f1_score(df['unstable_flight'], y_clf_pred, zero_division=1))

# Load hoặc Train Mô hình RQ2
if os.path.exists(rq2_path):
    reg_model = joblib.load(rq2_path)
else:
    X_reg = df[reg_features]
    y_reg = df['detection_confidence_avg']
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_reg, y_reg)

# Tính chỉ số RQ2
y_reg_pred = reg_model.predict(df[reg_features])
rq2_mae = float(mean_absolute_error(df['detection_confidence_avg'], y_reg_pred))
rq2_rmse = float(np.sqrt(mean_squared_error(df['detection_confidence_avg'], y_reg_pred)))

DRONE_IMAGES = [
    {"url": "https://images.unsplash.com/photo-1508614589041-895b88991e3e?w=800&q=80", "desc": "Drone thực hiện khảo sát không trung vùng pin năng lượng mặt trời"},
    {"url": "https://images.unsplash.com/photo-1509391365360-2e959784a276?w=800&q=80", "desc": "Camera nhiệt phát hiện điểm nóng (Hotspot) bất thường trên tấm pin"},
    {"url": "https://images.unsplash.com/photo-1613665813446-82a78c468a1d?w=800&q=80", "desc": "Soi chi tiết vết nứt bề mặt tấm pin bằng thuật toán thị giác"},
    {"url": "https://images.unsplash.com/photo-1548337138-e87d889cc369?w=800&q=80", "desc": "Kiểm tra hiện tượng che bóng (Shading/Bụi bẩn) giảm hiệu suất"}
]


# ==========================================
# 3. ENDPOINTS CHO FRONTEND UI
# ==========================================
@app.get("/api/v1/metrics")
def get_metrics():
    return {
        "rq1": {"accuracy": rq1_acc, "f1_score": rq1_f1},
        "rq2": {"mae": rq2_mae, "rmse": rq2_rmse},
        "dataset_size": len(df)
    }

@app.get("/api/v1/telemetry/{idx}")
def get_telemetry_sample(idx: int):
    start_time = time.time()
    
    idx = idx % len(df)
    current = df.iloc[idx].to_dict()
    
    # Dự báo RQ1 (An toàn bay)
    clf_in = pd.DataFrame([{
        'battery_level_pct': current.get('battery_level_pct', 100),
        'wind_speed_mps': current.get('wind_speed_mps', 0),
        'proximity_to_restricted_zone_m': current.get('proximity_to_restricted_zone_m', 1000),
        'ambient_temp_C': current.get('ambient_temp_C', 25)
    }])
    is_unstable = int(clf_model.predict(clf_in)[0])
    prob_risk = float(clf_model.predict_proba(clf_in)[0][1])
    
    # Dự báo RQ2 (Độ tin cậy Camera)
    reg_in = pd.DataFrame([{
        'ambient_temp_C': current.get('ambient_temp_C', 25),
        'wind_speed_mps': current.get('wind_speed_mps', 0),
        'detected_object_count': current.get('detected_object_count', 0),
        'thermal_signature_intensity': current.get('thermal_signature_intensity', 0)
    }])
    pred_conf = float(reg_model.predict(reg_in)[0])
    
    img_info = DRONE_IMAGES[idx % len(DRONE_IMAGES)]
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "sample_index": idx,
        "timestamp": str(current.get('timestamp', '')),
        "mission_id": str(current.get('mission_id', '')),
        "drone_id": str(current.get('drone_id', '')),
        "telemetry": {
            "battery_level_pct": float(current.get('battery_level_pct', 0)),
            "wind_speed_mps": float(current.get('wind_speed_mps', 0)),
            "ambient_temp_C": float(current.get('ambient_temp_C', 0)),
            "weather_condition": str(current.get('weather_condition', 'Clear')),
            "altitude_m": float(current.get('altitude_m', 0)),
            "distance_to_base_m": float(current.get('distance_to_base_m', 0)),
            "proximity_to_restricted_zone_m": float(current.get('proximity_to_restricted_zone_m', 0)),
            "gps_lat": float(current.get('gps_lat', 0.0)),
            "gps_lon": float(current.get('gps_lon', 0.0)),
            "detected_object_count": int(current.get('detected_object_count', 0))
        },
        "rq1_prediction": {
            "is_unstable": is_unstable,
            "risk_probability": prob_risk
        },
        "rq2_prediction": {
            "predicted_confidence": pred_conf
        },
        "image": img_info,
        "latency_ms": round(latency_ms, 2)
    }
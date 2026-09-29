# DATA DICTIONARY - DRONE TELEMETRY ANALYTICS & SOLAR PANEL INSPECTION

| Feature Name | Data Type | Measurement Unit | Business Meaning & Role | Analytical Mapping |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | DateTime | YYYY-MM-DD HH:MM:SS | Sensor recording timestamp | Time-series Index |
| `mission_id` / `drone_id` | String | N/A | Identifier for missions and UAV units | Metadata |
| `latitude` | Float | Decimal Degrees | Drone GPS latitude coordinate | Spatial Navigation |
| `longitude` | Float | Decimal Degrees | Drone GPS longitude coordinate | Spatial Navigation |
| `altitude` | Float | Meters (m) | Flight altitude above ground | Flight Safety |
| `imu_acc` | Float | m/s² | Inertial measurement unit acceleration | RQ1 (Stability) |
| `imu_gyro` | Float | rad/s or deg/s | Gyroscope angular rotation rate | RQ1 (Stability) |
| `imu_var` | Float | (m/s²)² | Accelerometer rolling variance (Imputed 9 NaNs) | RQ1 (Core Feature FE-01) |
| `speed_diff` | Float | m/s | First-order velocity step differential (Imputed 1 NaN) | RQ1 (Core Feature FE-02) |
| `wind_speed_mps` | Float | m/s | Ambient wind speed | RQ1 (Environmental Hazard) |
| `battery_level` | Float | Percentage (%) | Remaining battery state of charge | RQ3 (Energy) |
| `battery_drop` | Float | %/step | Instantaneous battery discharge rate | RQ3 (Core Feature FE-03) |
| `weather_condition` | Categorical | Clear/Cloudy/Rain/Fog | Local atmospheric conditions | RQ2 (Environmental Factor) |
| `ambient_temp_C` | Float | Celsius (°C) | Measured ambient temperature | RQ2 (Thermal Impact) |
| `thermal_signature_intensity` | Float | W/m² | Thermal radiation intensity of solar panels | RQ2 (Hotspot Defect Quality) |
| `detection_confidence_avg` | Float (0.0 - 1.0) | N/A | Defect detection confidence score (Target Continuous) | RQ2 (Regression Target) |
| `gps_anomaly` | Binary (0/1) | N/A | GPS drift / loss warning flag (93.70% occurrence) | Risk Feature |
| `unstable_battery` | Binary (0/1) | N/A | Premature battery failure indicator (37.96% occurrence) | RQ3 (Classification Target) |
| `unstable_wind` | Binary (0/1) | N/A | High wind turbulence indicator (28.72% occurrence) | RQ1 (Interaction Feature) |
| `unstable_flight` | Binary (0/1) | N/A | Flight instability status (0: Nominal, 1: Instability; 56.08%) | RQ1 (Primary Classification Target) |
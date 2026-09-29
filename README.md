# Drone Telemetry Analytics & Solar Panel Inspection

A quantitative, data-driven analytics framework and predictive modeling system designed to assess UAV flight safety risks and evaluate thermal payload inspection reliability across large-scale photovoltaic (PV) solar farms.

---

## 1. Academic Context & Course Information
- **Course**: DTA301 – Data Analytics (Research-Based Learning - RBL)
- **Project Title**: Drone Telemetry Analytics & Solar Panel Inspection
- **Supervisor**: Ngo Dang Ha An
- **Team**: Group 2
  - **SE193320** – Le Ngoc Hung (Analytics Lead)
  - **SE181676** – Do Tran Hai Son (Domain Modeling & Operations)
  - **SE180020** – Tran Phan Khai Hung (Data Engineering & Pipeline)
  - **SE180055** – Le Do Nhat Anh (Data Cleaning, Preprocessing & Data Dictionary)
  - **SE192292** – Tran Trieu Bao Long (Exploratory Data Analysis & Visualization)

---

## 2. Overview & Motivation

Autonomous Unmanned Aerial Vehicles (UAVs) equipped with thermal infrared and RGB payloads are widely adopted to detect hotspot anomalies and physical degradation in utility-scale solar farms. These operations continuously log dense sensor telemetry streams, including Inertial Measurement Unit (IMU) acceleration, angular rate, battery discharge metrics, GPS positional coordinates, and local environmental indicators.

However, field operations encounter severe stability and inspection fidelity bottlenecks:
- Aerodynamic disturbances and wind gusts introduce erratic IMU drift and positioning anomalies.
- Irregular battery depletion rates endanger mission completion and risk UAV loss.
- Ambient weather conditions (rain, fog, cloud cover) heavily degrade thermal radiation visibility and computer vision defect detection confidence.

Rather than relying on static, heuristic thresholds, the **Drone Telemetry Analytics & Solar Panel Inspection** project executes the **Data Analytics Lifecycle (Phases 1–6)** to extract engineered cross-features, build predictive classification and regression baselines, and translate statistical outputs into operational decision insights.

---

## 3. Problem Statement

During automated solar farm monitoring, flight operators struggle to maintain flight stability and verify inspection fidelity due to unpredictable weather variations, IMU sensor turbulence, high GPS noise, and sudden power dropouts.

Quantitative evidence from collected mission records reveals:
- **56.08%** of flight segments exhibit flight instability (`unstable_flight = 1`).
- **93.70%** encounter positioning drift or anomalies (`gps_anomaly = 1`).
- **37.96%** suffer from irregular battery depletion (`unstable_battery = 1`).
- Mean defect detection confidence (`detection_confidence_avg`) drops to **0.65 (65%)**, deteriorating significantly in non-nominal atmospheric conditions.

---

## 4. Research Questions (RQs)

- **Research Question 1 (RQ1 - Flight Instability Modeling)**: Which operational and environmental factors (`wind_speed_mps`, `imu_var`, `speed_diff`) have the strongest statistical impact on the likelihood of drone instability (`unstable_flight`) during solar panel scanning?
- **Research Question 2 (RQ2 - Payload Inspection Quality)**: To what degree do atmospheric conditions (`weather_condition`) and ambient temperature (`ambient_temp_C`) degrade object detection confidence (`detection_confidence_avg`) and thermal radiation intensity (`thermal_signature_intensity`)?
- **Research Question 3 (RQ3 - Predictive Battery Degradation)**: Can time-series features of battery drop rate (`battery_drop`) and GPS alerts (`gps_anomaly`) forecast premature power failure (`unstable_battery`) prior to critical failure points?

---

## 5. Dataset Architecture & Access

### 5.1. Raw Datasets (`data/raw/`)

Download the raw datasets and place them into the `data/raw/` directory:

| Dataset Name | Records / Dimensions | Download Link | Primary Analytical Function |
| :--- | :--- | :--- | :--- |
| `engineered_features_labels.csv` | 5,000 rows × 9 cols | [Download Link](https://drive.google.com/uc?export=download&id=16Fhb79Md7xeRd1UnvLZX1O_5ZcBXbUCJ) | Feature matrix for RQ1 & RQ3 classification models. |
| `uav_navigation_dataset.csv` | 5,000 rows × 15 cols | [Download Link](https://drive.google.com/uc?export=download&id=1GSmGcMDsWNvbr6G1NUO_TTte0ZzoMXKv) | Physical trajectory & motion analysis. |
| `SurveilDrone-Net23.csv` | 140,256 rows × 34 cols | [Download Link](https://drive.google.com/uc?export=download&id=1aOnszIUXpFpgeXhdIEjJn9m5ILmIQlxn) | Core dataset for RQ2 regression and ANOVA modeling. |

### 5.2. Processed Datasets (`data/processed/`)

| Dataset Name | Description |
| :--- | :--- |
| `engineered_features_cleaned.csv` | Imputed (1 missing in `speed_diff`, 9 missing in `imu_var`), type-cast, and verified feature matrix. |
| `SurveilDrone_Net23_cleaned.csv` | Cleaned environmental inspection dataset with physical boundary validation. |

---

## 6. End-to-End Data Analytics Pipeline

```text
Raw Telemetry Streams & Inspection Datasets
                     ↓
Phase 1: Discovery & Problem Formulation
(Business context, stakeholder mapping, RQ1-RQ3 formulation)
                     ↓
Phase 2: Data Preparation & Quality Assurance
(Missing value imputation, outlier detection, physical range validation, type casting)
                     ↓
Phase 3: Model Planning & Feature Engineering
(Extraction of imu_var, speed_diff, battery_drop; Data Dictionary establishment)
                     ↓
Phase 4: Model Building & Exploratory Data Analysis
(Correlation heatmaps, distribution skewness checks, hypothesis testing)
                     ↓
Phase 5: Model Evaluation & Validation
(Classification: AUC-ROC, F1-score; Regression: MAE, RMSE, Pearson r)
                     ↓
Phase 6: Operationalize & Decision Support
(ML-driven risk thresholds translated into dashboard recommendations)
```
---
## 7. Data Cleaning, Validation & Feature Engineering

### 7.1. Data Validation & Imputation (Week 4 Focus)
Time-Series Missing Imputation: Handled 1 missing observation in speed_diff and 9 missing observations in imu_var using forward linear interpolation, preserving physical motion continuity.

Operational Boundary Verification:

battery_level_pct constrained strictly to [0, 100%].

wind_speed_mps bounded to realistic thresholds ([0, 35 m/s]).

Spatial coordinates validated within legitimate solar installation geofences.

Type Standardization: Converted timestamp to standard datetime format; standardized binary targets (unstable_flight, gps_anomaly, unstable_battery) to integers (0/1).

### 7.2. Core Feature Set

| Feature Name | Type | Unit | Operational & Analytical Interpretation | Associated Target |
| :--- | :--- | :--- | :--- | :--- |
| `imu_var` | Float | (m/s2)2 | Rolling variance of accelerometer signals; measures turbulence shocks. | RQ1 (unstable_flight) |
| `speed_diff` | Float | m/s | First-order velocity step differential. | RQ1 (unstable_flight) |
| `battery_drop` | Float | %/step | Instantaneous discharge rate across consecutive time steps. | RQ3 (unstable_battery) |
| `wind_speed_mps` | Float | m/s | Ambient wind velocity recorded during flight segment. | RQ1 Covariate |
| `ambient_temp_C` | Float | deg C | Ambient field temperature. | RQ2 Covariate |
| `thermal_signature_intensity` | Float | W/m2 | Defect thermal radiation signal strength. | RQ2 Interaction |
| `detection_confidence_avg` | Float | [0.0, 1.0] | Mean confidence score output by defect detection model. | RQ2 Target |
| `unstable_flight` | Binary | 0 / 1 | Flight stability status (0: Nominal, 1: Instability event). | RQ1 Target |

---

## 8. Phase 6: Operationalize -- ML-Driven Decision Support

In the operational phase, trained predictive models generate real-time risk probabilities that drive actionable recommendations on the Fleet Analytics Dashboard:

| Operational Recommendation | Data Analytics & ML-Driven Trigger | Analytical Support Component |
| :--- | :--- | :--- |
| Emergency Landing | RQ1 model predicts instability risk P(unstable_flight) >= 0.85 or extreme battery drop rate detected. | High-Risk Anomaly Detection (RQ1 & RQ3) |
| Return to Base | RQ3 time-series model forecasts premature power depletion before route completion. | Predictive Energy Degradation (RQ3) |
| Delay / Postpone Mission | RQ2 regression model forecasts low detection confidence due to rain or fog. | Payload Inspection Quality Regression (RQ2) |
| Reroute / Adjust Altitude | Sensor variance imu_var spikes significantly under high crosswinds (> 12 m/s). | Multivariate Interaction Analysis (RQ1) |
| Deploy Backup UAV | Primary UAV is recalled early, leaving partial solar panel strings uninspected. | Fleet Efficiency Optimization |
| Continue Mission | Instability risk is low (P < 0.20) and forecasted detection confidence meets benchmark (>= 0.65). | Nominal Operational State |
---
## 9.Repository Structure
```text
DTA301_UAV_Analytics/
│
├── data/
│   ├── raw/                                 # Immutable raw datasets (download via section 5.1)
│   │   ├── engineered_features_labels.csv
│   │   ├── uav_navigation_dataset.csv
│   │   └── SurveilDrone-Net23.csv
│   ├── processed/                           # Cleaned, modeling-ready datasets
│   │   ├── engineered_features_cleaned.csv
│   │   └── SurveilDrone_Net23_cleaned.csv
│   └── data_dictionary.md                   # Full 20-feature academic dictionary
│
├── notebooks/
│   ├── 01_data_cleaning_nhatanh.ipynb       # Phase 2: Missing imputation, validation, type casting
│   ├── 02_eda_baolong.ipynb                 # Phase 2: EDA visualizations for RQ1, RQ2, RQ3
│   └── 03_baseline_modeling.ipynb           # Phase 3: Baseline models & metric evaluations
│
├── src/
│   ├── __init__.py
│   ├── config.py                            # Reproducibility seed (42) and system paths
│   ├── data_cleaning.py                     # Reusable imputation and cleaning routines
│   └── visualization.py                     # Custom plotting scripts for distributions & heatmaps
│
├── reports/
│   ├── figures/                             # High-resolution generated charts (.png)
│   │   ├── rq1_unstable_flight_dist.png
│   │   ├── rq2_weather_vs_confidence.png
│   │   ├── rq3_battery_drop_series.png
│   │   └── correlation_heatmap.png
│   └── EDA_Draft_Report_Week4.md            # Week 4 deliverable summary
│
├── .gitignore                               # Git ignore rules for virtual environments & raw CSVs
├── requirements.txt                         # Pinned dependencies for reproducible execution
└── README.md                                # Project overview and academic documentation
```
---
## 10. Reproducibility & Setup Guide
This project strictly adheres to the course Reproducibility Framework:

Global random seed fixed at random_seed = 42 across all scripts.

Consistent 80/20 Train-Test split for model evaluation.

Execution Steps:
Clone the repository:
git clone https://github.com/Win-NA/DTA301_UAV_Analytics.git
cd DTA301_UAV_Analytics

Install pinned dependencies:
pip install -r requirements.txt

Download Raw Data:
Download the datasets from the links in Section 5.1 and place them inside the data/raw/ folder.

Execute Data Cleaning (Nhat Anh - Task Week 4):
Open and execute notebooks/01_data_cleaning_nhatanh.ipynb to clean missing values, enforce operational boundaries, and export to data/processed/.

Execute Exploratory Data Analysis (Bao Long - Task Week 4):
Open and execute notebooks/02_eda_baolong.ipynb to inspect distributions, generate correlation heatmaps, and export figures to reports/figures/.
---
## 11. Key Deliverables & Expected Outputs
Cleaned Telemetry & Inspection Datasets: Fully imputed, validated, and normalized tabular data ready for statistical learning.

Academic Data Dictionary: Formal reference document defining 20 features, physical units, and research question mapping.

EDA Visualizations: Comprehensive distribution plots and correlation heatmaps aligned with RQ1-RQ3.

Predictive & Prescriptive Baseline: Documented modeling foundation ready for Phase 3 (Deliverable D2).

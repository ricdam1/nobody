# ⚙️ MotorGuard — Predictive Maintenance for Industrial Motors

An end-to-end machine learning system that classifies the health status
of industrial motors in real time using operational sensor data.

Built with Python, XGBoost, SHAP, MLflow, and Streamlit.

---

## Problem Statement

Industrial motors fail without warning — costing manufacturers billions
annually in unplanned downtime. Most facilities still rely on reactive
or scheduled maintenance, both of which are wasteful and unreliable.

MotorGuard monitors motor sensor readings continuously and classifies
each motor into one of four health states before failure occurs.

---

## Health Classification System

| Status | Meaning | Action |
|---|---|---|
| 🟢 Normal | Operating within safe parameters | No action required |
| 🟡 Early Warning | Early stress signals detected | Schedule inspection |
| 🟠 Degraded | Performance degrading | Maintenance within 48hrs |
| 🔴 Critical | Failure imminent | Immediate action required |

---

##  Dataset

**AI4I 2020 Predictive Maintenance Dataset**
- Source: Kaggle / UCI Machine Learning Repository
- Size: 10,000 rows × 14 columns
- License: CC BY 4.0
- Type: Synthetic dataset modeled on real industrial data

```python
import kagglehub
path = kagglehub.dataset_download(
    "shivamb/machine-predictive-maintenance-classification"
)
```

---

## Feature Engineering

Three physics-based features engineered on top of raw sensor data:

| Feature | Formula | Why It Matters |
|---|---|---|
| Power_W | (2π × RPM × Torque) / 60 | Power failure triggers below 3500W or above 9000W |
| Temp_Delta | Process Temp − Air Temp | Should always be ~10K — deviation signals heat failure |
| Wear_Torque | Tool Wear × Torque | Exceeding type limit triggers overstrain failure |

---

##  Models Trained

| Model | F1 Macro | Recall (Critical) | Business Cost |
|---|---|---|---|
| Random Forest | 0.8243 | 0.72 | 88 |
| **XGBoost** ✅ | **0.8386** | **0.76** | **76** |
| LightGBM | 0.8729 | 0.72 | 80 |

**Winner: XGBoost** — lowest business cost and highest Critical recall.

Business cost weights missed Critical failures at 10× the cost of
false alarms — reflecting real-world maintenance priorities.

---
##  Project Structure
```bash
motorguard/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw_data.csv
│   └── processed_data.csv
│
├── notebooks/
│   ├── plot1_class_distribution.png
│   ├── plot2_correlation_heatmap.png
│   ├── plot3_feature_distributions.png
│   ├── plot4_smote_effect.png
│   ├── plot5_confusion_matrices.png
│   ├── plot6_business_cost.png
│   └── plot7_shap_importance.png
│
├── src/
│   ├── eda/
│   │   └── preprocess.ipynb
│   │
│   ├── mlruns/
│   │
│   └── train.ipynb
│
├── .gitignore
├── check_setup.py
└── README.md
```
##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/motorguard.git
cd motorguard
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\Activate.ps1    # Windows PowerShell
source venv/bin/activate      # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebooks in order


### 5. Launch the Streamlit app

```bash
streamlit run app\streamlit_app.py
```

Open your browser at `http://localhost:8501`

---

## How SHAP Explainability Works

Every prediction comes with a SHAP (SHapley Additive exPlanations)
bar chart that shows which features drove the classification and by
how much.

- 🔴 Red bars — features pushing toward the predicted class
- 🟢 Green bars — features pushing away from the predicted class

This gives maintenance engineers a reason to trust the alert,
not just a black-box verdict.

---

## Quick Test Scenarios

| Scenario | Type | Air Temp | Proc Temp | RPM | Torque | Tool Wear | Expected |
|---|---|---|---|---|---|---|---|
| Healthy motor | M | 298K | 310K | 1500 | 40Nm | 50min | 🟢 Normal |
| Stress building | M | 300K | 310K | 1300 | 65Nm | 180min | 🟡 Early Warning |
| Heat failure | M | 305K | 305K | 1300 | 60Nm | 150min | 🟠 Degraded |
| Overstrain | L | 300K | 313K | 1200 | 70Nm | 240min | 🔴 Critical |

---

## Experiment Tracking

All training runs are logged with MLflow.

To view the experiment dashboard:

```bash
mlflow ui
```

Open `http://localhost:5000` in your browser.

---

## Known Limitations

- Dataset is synthetic — not collected from real physical sensors
- Model is trained on one motor type configuration
- Vibration and current waveform data not included
- Designed as a proof-of-concept pipeline

## Future Improvements

- Swap in real sensor data
- Add FastAPI inference microservice
- Containerize with Docker
- Add Evidently drift monitoring
- Add motor health trend history over time

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data | AI4I 2020 via kagglehub |
| Preprocessing | pandas, scikit-learn, imbalanced-learn |
| Modelling | XGBoost, LightGBM, Random Forest |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Dashboard | Streamlit |
| Serialization | joblib |

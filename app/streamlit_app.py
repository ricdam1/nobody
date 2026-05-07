# app/streamlit_app.py
# MotorGuard — Predictive Maintenance Dashboard

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="MotorGuard",
    page_icon="⚙️",
    layout="wide"
)

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")

@st.cache_resource
def load_artifacts():
    model        = joblib.load(os.path.join(ARTIFACTS_DIR, "motorguard_model.pkl"))
    scaler       = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    le           = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_cols.pkl"))
    explainer    = joblib.load(os.path.join(ARTIFACTS_DIR, "shap_explainer.pkl"))
    return model, scaler, le, feature_cols, explainer

model, scaler, le, feature_cols, explainer = load_artifacts()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("⚙️ MotorGuard")
st.markdown("### Predictive Maintenance Dashboard for Industrial Motors")
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR — MOTOR INPUTS
# ─────────────────────────────────────────────

st.sidebar.title("🔧 Motor Readings")
st.sidebar.markdown("Adjust the sliders to simulate motor sensor readings.")

product_type = st.sidebar.selectbox(
    "Product Type",
    options=["L — Low", "M — Medium", "H — High"],
    index=1
)
type_map     = {"L — Low": 1, "M — Medium": 2, "H — High": 0}
type_encoded = type_map[product_type]

air_temp = st.sidebar.slider(
    "Air Temperature (K)",
    min_value=295.0,
    max_value=305.0,
    value=298.0,
    step=0.1
)

process_temp = st.sidebar.slider(
    "Process Temperature (K)",
    min_value=305.0,
    max_value=315.0,
    value=310.0,
    step=0.1
)

rpm = st.sidebar.slider(
    "Rotational Speed (RPM)",
    min_value=1168,
    max_value=2886,
    value=1500,
    step=10
)

torque = st.sidebar.slider(
    "Torque (Nm)",
    min_value=3.8,
    max_value=76.6,
    value=40.0,
    step=0.1
)

tool_wear = st.sidebar.slider(
    "Tool Wear (min)",
    min_value=0,
    max_value=253,
    value=100,
    step=1
)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING — same as training
# ─────────────────────────────────────────────

power_w     = (2 * np.pi * rpm * torque) / 60
temp_delta  = process_temp - air_temp
wear_torque = tool_wear * torque

# Build input in exact same column order as training
input_data = pd.DataFrame([{
    "Type_encoded":              type_encoded,
    "Air temperature [K]":       air_temp,
    "Process temperature [K]":   process_temp,
    "Rotational speed [rpm]":    rpm,
    "Torque [Nm]":               torque,
    "Tool wear [min]":           tool_wear,
    "Power_W":                   power_w,
    "Temp_Delta":                temp_delta,
    "Wear_Torque":               wear_torque
}])[feature_cols]

# Scale
input_scaled = scaler.transform(input_data)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

pred_encoded = model.predict(input_scaled)[0]
pred_label   = le.inverse_transform([pred_encoded])[0]
pred_proba   = model.predict_proba(input_scaled)[0]
confidence   = pred_proba.max() * 100

# ─────────────────────────────────────────────
# STATUS CONFIG
# ─────────────────────────────────────────────

status_config = {
    "Normal": {
        "emoji":   "🟢",
        "color":   "#2ecc71",
        "message": "Motor is operating normally. No action required.",
        "bg":      "#d5f5e3"
    },
    "Early Warning": {
        "emoji":   "🟡",
        "color":   "#f1c40f",
        "message": "Early signs of stress detected. Schedule inspection soon.",
        "bg":      "#fef9e7"
    },
    "Degraded": {
        "emoji":   "🟠",
        "color":   "#e67e22",
        "message": "Motor performance is degraded. Maintenance recommended.",
        "bg":      "#fdebd0"
    },
    "Critical": {
        "emoji":   "🔴",
        "color":   "#e74c3c",
        "message": "⚠️ Critical failure imminent. Immediate action required!",
        "bg":      "#fadbd8"
    }
}

cfg = status_config[pred_label]

# ─────────────────────────────────────────────
# MAIN LAYOUT — THREE COLUMNS
# ─────────────────────────────────────────────

col1, col2, col3 = st.columns([1.5, 1.5, 2])

# ── Column 1: Health Status ──
with col1:
    st.markdown("### 🏥 Health Status")
    st.markdown(
        f"""
        <div style="
            background-color: {cfg['bg']};
            border-left: 6px solid {cfg['color']};
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        ">
            <h1 style="color:{cfg['color']}; margin:0;">
                {cfg['emoji']} {pred_label}
            </h1>
            <p style="font-size:16px; margin-top:10px;">
                {cfg['message']}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(f"**Confidence:** `{confidence:.1f}%`")

    # ── Alert banner ──
    if pred_label == "Critical":
        st.error("🚨 ALERT: Schedule immediate maintenance!")
    elif pred_label == "Degraded":
        st.warning("⚠️ WARNING: Plan maintenance within 48 hours.")
    elif pred_label == "Early Warning":
        st.info("ℹ️ INFO: Monitor closely. Inspect within the week.")
    else:
        st.success("✅ All systems normal.")

# ── Column 2: Computed Values & Probabilities ──
with col2:
    st.markdown("### ⚙️ Computed Values")

    # Power health indicator
    power_status = ""
    if power_w < 3500:
        power_status = "⚠️ Too Low"
    elif power_w > 9000:
        power_status = "⚠️ Too High"
    else:
        power_status = "✅ Normal Range"

    st.metric(
        "Power Output",
        f"{power_w:.0f} W",
        delta=power_status
    )
    st.metric(
        "Temperature Delta",
        f"{temp_delta:.2f} K",
        delta="Normal" if 8 <= temp_delta <= 12 else "⚠️ Abnormal"
    )
    st.metric(
        "Wear × Torque",
        f"{wear_torque:.0f}",
        delta="⚠️ High Risk" if wear_torque > 11000 else "✅ Safe"
    )

    st.markdown("---")
    st.markdown("**Class Probabilities:**")

    # Sort by probability descending
    proba_df = pd.DataFrame({
        "Class":       le.classes_,
        "Probability": (pred_proba * 100).round(1)
    }).sort_values("Probability", ascending=False)

    for _, row in proba_df.iterrows():
        st.progress(
            int(row["Probability"]),
            text=f"{row['Class']}: {row['Probability']}%"
        )

# ── Column 3: SHAP Explanation ──
with col3:
    st.markdown("### 🔍 Why This Classification?")
    st.markdown("*Which features drove this prediction the most*")

    try:
        shap_vals = explainer.shap_values(input_scaled)

        # Handle both list and 3D array SHAP output formats
        if isinstance(shap_vals, list):
            # List of arrays — one per class
            shap_for_class = shap_vals[pred_encoded][0]
        elif shap_vals.ndim == 3:
            # 3D array — shape (samples, features, classes)
            shap_for_class = shap_vals[0, :, pred_encoded]
        else:
            # 2D fallback
            shap_for_class = shap_vals[0]

        shap_df = pd.DataFrame({
            "Feature":    feature_cols,
            "SHAP Value": shap_for_class
        }).sort_values("SHAP Value", key=abs, ascending=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = [
            "#e74c3c" if v > 0 else "#2ecc71"
            for v in shap_df["SHAP Value"]
        ]
        ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on prediction)")
        ax.set_title(f"Feature Impact — {pred_label}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption(
            "🔴 Red = pushes toward this classification  |  "
            "🟢 Green = pushes away from this classification"
        )

    except Exception as e:
        st.warning(f"SHAP chart unavailable: {e}")

# ─────────────────────────────────────────────
# BOTTOM — FAILURE RISK INDICATORS
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown("### 🧪 Failure Risk Indicators")

risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

with risk_col1:
    power_risk = "🔴 High" if power_w < 3500 or power_w > 9000 else "🟢 Low"
    st.markdown(f"**Power Failure Risk**")
    st.markdown(f"{power_risk}")
    st.caption(f"Safe range: 3500–9000 W\nYours: {power_w:.0f} W")

with risk_col2:
    heat_risk = "🔴 High" if temp_delta < 8 else "🟢 Low"
    st.markdown(f"**Heat Dissipation Risk**")
    st.markdown(f"{heat_risk}")
    st.caption(f"Safe range: ≥ 8K delta\nYours: {temp_delta:.2f} K")

with risk_col3:
    strain_limits = {"L — Low": 11000, "M — Medium": 12000, "H — High": 13000}
    strain_limit  = strain_limits[product_type]
    strain_risk   = "🔴 High" if wear_torque > strain_limit else "🟢 Low"
    st.markdown(f"**Overstrain Risk**")
    st.markdown(f"{strain_risk}")
    st.caption(f"Limit for {product_type}: {strain_limit}\nYours: {wear_torque:.0f}")

with risk_col4:
    wear_risk = "🔴 High" if tool_wear > 200 else "🟡 Medium" if tool_wear > 150 else "🟢 Low"
    st.markdown(f"**Tool Wear Risk**")
    st.markdown(f"{wear_risk}")
    st.caption(f"Your wear: {tool_wear} min")

# ─────────────────────────────────────────────
# BOTTOM — INPUT SUMMARY TABLE
# ─────────────────────────────────────────────

st.markdown("---")
with st.expander("📋 View Full Input Summary"):
    st.dataframe(
        input_data.T.rename(columns={0: "Value"}),
        use_container_width=True
    )

st.markdown("---")
st.caption(
    "MotorGuard v1.0 — AI-Powered Predictive Maintenance | "
    "Built with Streamlit + XGBoost + SHAP"
)
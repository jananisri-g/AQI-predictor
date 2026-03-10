import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import date, time
from model import predict_aqi, suggest_precautions, get_actual_vs_predicted, LOCATION_FILES

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌫️",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
AQI_BANDS = [
    (50,  "#4CAF50", "Good"),
    (100, "#F9D65C", "Satisfactory"),
    (200, "#F4A261", "Moderate"),
    (300, "#E76F51", "Poor"),
    (400, "#A78BFA", "Very Poor"),
    (500, "#C08497", "Severe"),
]

LOCATION_LIST = list(LOCATION_FILES.keys())   # ["Delhi", "Bengaluru"]


def get_aqi_color(aqi):
    for threshold, color, _ in AQI_BANDS:
        if aqi <= threshold:
            return color
    return "#C08497"


def get_aqi_label(aqi):
    for threshold, _, label in AQI_BANDS:
        if aqi <= threshold:
            return label
    return "Severe"


def normalise_precautions(precautions):
    """Accept dict or legacy flat list, always return dict."""
    if isinstance(precautions, dict):
        return precautions
    prec_dict = {}
    current_group = "General"
    GROUP_KEYWORDS = ("(", "Patients", "Workers", "Adults", "Women", "Elderly", "Teenagers")
    for item in precautions:
        item = item.strip()
        if not item or item == "\n":
            continue
        if any(kw in item for kw in GROUP_KEYWORDS) and len(item) < 60:
            current_group = item
            prec_dict.setdefault(current_group, [])
        else:
            clean = item.replace("\n\n", "").strip()
            if clean:
                prec_dict.setdefault(current_group, []).append(clean)
    return prec_dict


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/air-quality.png", width=64)
    st.title("AQI System")
    st.caption("Air Quality Prediction & Health Advisory")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠  Prediction", "📊  Model Performance", "📖  Methodology"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**AQI Reference Scale**")
    for _, color, label in AQI_BANDS:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0'>"
            f"<div style='width:14px;height:14px;border-radius:3px;background:{color}'></div>"
            f"<span style='font-size:0.85rem'>{label}</span></div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("Final Year Project · 2025")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1 — PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
if "Prediction" in page:

    st.title("🌫️ AQI Prediction & Health Advisory")
    st.write(
        "Select a **location**, pick a **date** from the calendar, choose the **hour**, "
        "then click **Predict AQI** to receive group-specific health recommendations."
    )
    st.divider()

    # ── Input row ─────────────────────────────────────────────────────────────
    col_loc, col_date, col_hour, col_btn = st.columns([1.5, 2, 1.5, 1])

    with col_loc:
        st.markdown("**📍 Location**")
        selected_location = st.selectbox(
            "Location",
            options=LOCATION_LIST,
            index=0,
            label_visibility="collapsed",
        )

    with col_date:
        st.markdown("**📅 Date**")
        selected_date = st.date_input(
            "Date",
            value=date(2025, 1, 20),
            label_visibility="collapsed",
        )

    with col_hour:
        st.markdown("**🕐 Hour of Day**")
        # Clock-style slider — 0–23 with AM/PM display
        selected_hour = st.slider(
            "Hour",
            min_value=0,
            max_value=23,
            value=15,
            format="%d:00",
            label_visibility="collapsed",
        )
        # Human-readable AM/PM label
        hour_label = (
            "12:00 noon" if selected_hour == 12
            else f"{selected_hour}:00 {'AM' if selected_hour < 12 else 'PM'}"
        )
        st.markdown(
            f"<div style='text-align:center;font-size:1.05rem;font-weight:600;"
            f"color:#6366f1;margin-top:4px'>🕐 {hour_label}</div>",
            unsafe_allow_html=True,
        )

    with col_btn:
        st.markdown("&nbsp;")   # vertical alignment spacer
        st.markdown("&nbsp;")
        predict_clicked = st.button("🔍 Predict AQI", use_container_width=True)

    # ── Assembled datetime preview ─────────────────────────────────────────────
    assembled_dt = f"{selected_date} {selected_hour:02d}:00:00"
    st.markdown(
        f"<div style='margin:8px 0 16px 0; padding:10px 16px; background:#f1f5f9; "
        f"border-radius:8px; font-size:0.9rem; color:#475569;'>"
        f"🗓️ <b>Selected:</b> &nbsp; {selected_location} &nbsp;·&nbsp; "
        f"{selected_date.strftime('%A, %d %B %Y')} &nbsp;·&nbsp; {hour_label}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Prediction ─────────────────────────────────────────────────────────────
    if predict_clicked:
        with st.spinner(f"Running prediction for {selected_location}…"):
            try:
                predicted_aqi = predict_aqi(assembled_dt, location=selected_location)
                category, precautions = suggest_precautions(predicted_aqi)
                color = get_aqi_color(predicted_aqi)

                # ── KPI metrics ───────────────────────────────────────────────
                st.divider()
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("📍 Location",    selected_location)
                k2.metric("Predicted AQI",  f"{round(predicted_aqi, 2)}")
                k3.metric("Category",        category)
                k4.metric("Risk Level",      get_aqi_label(predicted_aqi))

                # ── Gauge ─────────────────────────────────────────────────────
                st.subheader("AQI Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_aqi,
                    title={"text": f"Air Quality Index — {selected_location}", "font": {"size": 18}},
                    number={"font": {"size": 48, "color": color}},
                    gauge={
                        "axis": {
                            "range": [0, 500],
                            "tickvals": [0, 50, 100, 200, 300, 400, 500],
                            "ticktext": ["0", "50", "100", "200", "300", "400", "500"],
                        },
                        "bar": {"color": "#1f2937", "thickness": 0.2},
                        "steps": [
                            {"range": [0,   50],  "color": "#4CAF50"},
                            {"range": [50,  100], "color": "#F9D65C"},
                            {"range": [100, 200], "color": "#F4A261"},
                            {"range": [200, 300], "color": "#E76F51"},
                            {"range": [300, 400], "color": "#A78BFA"},
                            {"range": [400, 500], "color": "#C08497"},
                        ],
                        "threshold": {
                            "line": {"color": "#111", "width": 5},
                            "thickness": 0.75,
                            "value": predicted_aqi,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    height=380,
                    margin=dict(l=30, r=30, t=60, b=20),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # ── Health Risk Bar ───────────────────────────────────────────
                st.subheader("Health Risk Breakdown")
                risk_labels = ["Safe Margin", "Moderate Risk", "High Risk"]
                risk_values = [
                    round(max(0, 150 - predicted_aqi), 1),
                    round(predicted_aqi / 2, 1),
                    round(predicted_aqi / 3, 1),
                ]
                fig_risk = go.Figure(go.Bar(
                    x=risk_labels,
                    y=risk_values,
                    marker_color=["#4CAF50", "#F4A261", "#E76F51"],
                    text=risk_values,
                    textposition="outside",
                    width=0.45,
                ))
                fig_risk.update_layout(
                    yaxis_title="Relative Risk Score",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=340,
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
                )
                st.plotly_chart(fig_risk, use_container_width=True)

                # ── Precautions Table ─────────────────────────────────────────
                st.subheader("Recommended Precautions by Group")
                prec_dict = normalise_precautions(precautions)

                col_h1, col_h2 = st.columns([1, 2])
                col_h1.markdown("**Population Group**")
                col_h2.markdown("**Recommended Measures**")
                st.divider()

                for group, advice_list in prec_dict.items():
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.markdown(
                            f"<span style='font-weight:700;color:{color}'>{group}</span>",
                            unsafe_allow_html=True,
                        )
                    with c2:
                        for advice in advice_list:
                            st.markdown(f"• {advice}")
                    st.divider()

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2 — MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────
elif "Performance" in page:

    st.title("📊 Model Performance")
    st.write("Evaluation of the XGBoost model on the held-out test set (last 20% of data).")
    st.divider()

    # Location selector for performance page
    perf_location = st.selectbox(
        "Select location to evaluate:",
        options=LOCATION_LIST,
        index=0,
    )

    with st.spinner(f"Loading model results for {perf_location}…"):
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            actual, predicted = get_actual_vs_predicted(location=perf_location)

            mae  = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            r2   = r2_score(actual, predicted)
            mape = np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1, actual))) * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE",  f"{mae:.2f}",   help="Mean Absolute Error — lower is better")
            m2.metric("RMSE", f"{rmse:.2f}",  help="Root Mean Squared Error")
            m3.metric("R²",   f"{r2:.4f}",    help="Coefficient of determination — closer to 1.0 is better")
            m4.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")

            st.divider()

            # ── Actual vs Predicted line chart ────────────────────────────────
            st.subheader("Actual vs Predicted AQI")
            n_points = len(actual)
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=list(range(n_points)), y=actual.tolist(),
                mode="lines", name="Actual AQI",
                line=dict(color="#1d4ed8", width=1.8),
            ))
            fig_line.add_trace(go.Scatter(
                x=list(range(n_points)), y=predicted.tolist(),
                mode="lines", name="Predicted AQI",
                line=dict(color="#f97316", width=1.8, dash="dash"),
            ))
            fig_line.update_layout(
                xaxis_title="Test Data Points",
                yaxis_title="AQI",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=420,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # ── Residuals distribution ────────────────────────────────────────
            st.subheader("Residuals Distribution")
            residuals = actual - predicted
            fig_hist = px.histogram(
                x=residuals,
                nbins=60,
                labels={"x": "Residual (Actual − Predicted)", "y": "Count"},
                color_discrete_sequence=["#6366f1"],
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="#e76f51", line_width=2)
            fig_hist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=340,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption("A residuals distribution centred near zero indicates low systematic bias.")

            # ── Scatter: Actual vs Predicted ──────────────────────────────────
            st.subheader("Prediction Accuracy Scatter")
            fig_scatter = px.scatter(
                x=actual, y=predicted,
                labels={"x": "Actual AQI", "y": "Predicted AQI"},
                opacity=0.5,
                color_discrete_sequence=["#1d4ed8"],
            )
            min_val = float(min(actual.min(), predicted.min()))
            max_val = float(max(actual.max(), predicted.max()))
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode="lines", name="Perfect Prediction",
                line=dict(color="#e76f51", dash="dash", width=2),
            ))
            fig_scatter.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
                xaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Points close to the dashed diagonal indicate accurate predictions.")

        except Exception as e:
            st.error(f"Error loading model data: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3 — METHODOLOGY / ABOUT
# ──────────────────────────────────────────────────────────────────────────────
elif "Methodology" in page:

    st.title("📖 Methodology & About")
    st.divider()

    st.subheader("Project Overview")
    st.write(
        "This system predicts the Air Quality Index (AQI) at a given datetime and location "
        "using a machine learning model trained on historical AQI observations. Based on the "
        "predicted AQI, the system provides evidence-based health recommendations tailored to "
        "different population groups including children, elderly individuals, pregnant women, "
        "athletes, and patients with chronic respiratory or cardiovascular conditions."
    )

    st.divider()
    st.subheader("Supported Locations")
    loc_data = {
        "City":        list(LOCATION_FILES.keys()),
        "Data File":   list(LOCATION_FILES.values()),
    }
    st.dataframe(pd.DataFrame(loc_data), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Dataset")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**Source**")
        st.write("Historical AQI readings stored per city in `.xlsx` files.")
        st.markdown("**Format**")
        st.write(
            "Wide format — rows are dates, columns are hourly time slots. "
            "Melted to long format during preprocessing."
        )
    with col_d2:
        st.markdown("**Preprocessing Steps**")
        for step in [
            "Melt wide → long format",
            "Parse Datetime from Date + Time columns",
            "Coerce non-numeric AQI values to NaN and drop",
            "Sort chronologically and reset index",
        ]:
            st.markdown(f"• {step}")

    st.divider()
    st.subheader("Feature Engineering")
    st.write(
        "The model relies entirely on time-series derived features — no external weather or "
        "geographic data is required. The following features were constructed:"
    )
    feature_data = {
        "Feature": [
            "hour", "day", "dayofweek",
            "lag1", "lag2",
            "lag24", "lag48", "lag72",
            "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
        ],
        "Type": [
            "Temporal", "Temporal", "Temporal",
            "Lag", "Lag",
            "Lag", "Lag", "Lag",
            "Rolling", "Rolling", "Rolling",
        ],
        "Description": [
            "Hour of day (0–23)",
            "Day of month (1–31)",
            "Day of week (0 = Monday)",
            "AQI 1 hour prior",
            "AQI 2 hours prior",
            "AQI 24 hours prior (same time yesterday)",
            "AQI 48 hours prior",
            "AQI 72 hours prior",
            "3-hour rolling mean AQI",
            "6-hour rolling mean AQI",
            "12-hour rolling mean AQI",
        ],
    }
    st.dataframe(pd.DataFrame(feature_data), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Model")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**Algorithm:** XGBoost Regressor")
        st.markdown("**Train/Test Split:** 80% / 20% (chronological)")
        st.markdown("**Evaluation Metrics:** MAE, RMSE, R², MAPE")
        st.markdown("**Model Caching:** Each city's model is trained once and cached.")
    with col_m2:
        st.markdown("**Hyperparameters**")
        for k, v in {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
        }.items():
            st.markdown(f"• `{k}` = `{v}`")

    st.divider()
    st.subheader("AQI Health Classification")
    band_data = {
        "AQI Range":  ["0–50", "51–100", "101–200", "201–300", "301–400", "401–500"],
        "Category":   ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
        "Primary Health Concern": [
            "Minimal impact",
            "Minor breathing discomfort for sensitive individuals",
            "Breathing discomfort for people with lung/heart disease",
            "Breathing discomfort for most people on prolonged exposure",
            "Respiratory illness on prolonged exposure",
            "Serious respiratory and cardiovascular effects",
        ],
    }
    st.dataframe(pd.DataFrame(band_data), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Limitations")
    for lim in [
        "The model does not incorporate real-time meteorological data (wind speed, humidity, temperature), which are known contributors to AQI variation.",
        "Predictions are based solely on historical AQI patterns; sudden pollution events (industrial accidents, wildfires) cannot be anticipated.",
        "The 80/20 chronological split means the model was not evaluated against unseen seasonal periods outside the training window.",
        "Health advisory thresholds follow general Indian AQI classification guidelines and may not reflect local regulatory standards in all regions.",
        "Lag features require at least 72 hours of prior historical data, limiting predictions near the start of the dataset.",
    ]:
        st.markdown(f"⚠️ {lim}")

    st.divider()
    st.subheader("Technologies Used")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("**Machine Learning**")
        st.markdown("• XGBoost\n• scikit-learn\n• pandas / NumPy")
    with t2:
        st.markdown("**Visualisation**")
        st.markdown("• Plotly\n• Streamlit")
    with t3:
        st.markdown("**Environment**")
        st.markdown("• Python 3.x\n• Streamlit web app\n• Excel datasets (.xlsx) per city")

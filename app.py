"""
Streamlit dashboard for Sales Forecasting project.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("📈 Sales Forecasting Dashboard")
st.markdown("Basic-level time-series forecasting project using **Facebook Prophet**.")

# -----------------------------
# Load model and data
# -----------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("data/sales.csv", parse_dates=["date"])
    df = df[(df["store"] == 1) & (df["item"] == 1)][["date", "sales"]]
    df = df.rename(columns={"date": "ds", "sales": "y"})
    return df.sort_values("ds").reset_index(drop=True)

model = load_model()
history = load_data()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Forecast Settings")
horizon = st.sidebar.slider("Days to forecast into the future", min_value=7, max_value=90, value=30, step=7)

# -----------------------------
# Generate forecast
# -----------------------------
future = model.make_future_dataframe(periods=horizon)
forecast = model.predict(future)

# -----------------------------
# Key stats
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Historical Avg Daily Sales", f"{history['y'].mean():.0f}")
col2.metric("Last Recorded Sales", f"{history['y'].iloc[-1]:.0f}")
col3.metric(f"Predicted Avg (next {horizon} days)", f"{forecast['yhat'].iloc[-horizon:].mean():.0f}")

# -----------------------------
# Main chart
# -----------------------------
st.subheader("Sales History + Forecast")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(history["ds"], history["y"], label="Historical Sales", color="steelblue")
ax.plot(forecast["ds"].iloc[-horizon:], forecast["yhat"].iloc[-horizon:], label="Forecast", color="orange")
ax.fill_between(
    forecast["ds"].iloc[-horizon:],
    forecast["yhat_lower"].iloc[-horizon:],
    forecast["yhat_upper"].iloc[-horizon:],
    color="orange", alpha=0.2, label="Confidence Interval"
)
ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
st.pyplot(fig)

# -----------------------------
# Forecast table
# -----------------------------
st.subheader("Forecast Data Table")
display_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].iloc[-horizon:].copy()
display_df.columns = ["Date", "Predicted Sales", "Lower Bound", "Upper Bound"]
display_df["Date"] = display_df["Date"].dt.date
st.dataframe(display_df.round(1), use_container_width=True)

# -----------------------------
# Model evaluation (static, from forecast.py run)
# -----------------------------
st.subheader("Model Evaluation (on held-out last 30 days)")
try:
    eval_df = pd.read_csv("outputs/evaluation_results.csv")
    st.table(eval_df)
except FileNotFoundError:
    st.info("Run forecast.py first to generate evaluation results.")

st.markdown("---")
st.caption("Basic-level Time-Series Sales Forecasting project | Model: Facebook Prophet")

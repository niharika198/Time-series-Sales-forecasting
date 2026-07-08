"""
Time-Series Sales Forecasting - Basic Level Project
------------------------------------------------------
Steps:
1. Load data
2. EDA (trend, weekly pattern plots)
3. Train/test split (last 30 days as test)
4. Baseline model: naive forecast (yesterday's value repeated / seasonal naive)
5. Main model: Facebook Prophet
6. Evaluate both with MAE, RMSE, MAPE
7. Save trained model + plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

os.makedirs("outputs", exist_ok=True)

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("data/sales.csv", parse_dates=["date"])
df = df[(df["store"] == 1) & (df["item"] == 1)][["date", "sales"]]
df = df.rename(columns={"date": "ds", "sales": "y"})
df = df.sort_values("ds").reset_index(drop=True)

print(f"Loaded {len(df)} rows from {df.ds.min().date()} to {df.ds.max().date()}")

# -----------------------------
# 2. EDA
# -----------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(df["ds"], df["y"])
axes[0].set_title("Daily Sales Over Time")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Sales")

weekly_avg = df.copy()
weekly_avg["day_of_week"] = weekly_avg["ds"].dt.day_name()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekly_avg = weekly_avg.groupby("day_of_week")["y"].mean().reindex(order)

axes[1].bar(weekly_avg.index, weekly_avg.values, color="steelblue")
axes[1].set_title("Average Sales by Day of Week")
axes[1].set_ylabel("Avg Sales")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("outputs/eda_plots.png", dpi=100)
print("Saved EDA plots -> outputs/eda_plots.png")

# -----------------------------
# 3. Train/test split (last 30 days as test)
# -----------------------------
TEST_DAYS = 30
train = df.iloc[:-TEST_DAYS].copy()
test = df.iloc[-TEST_DAYS:].copy()

print(f"Train: {len(train)} rows | Test: {len(test)} rows")

# -----------------------------
# 4. Baseline model: seasonal naive (value from 7 days ago)
# -----------------------------
naive_preds = df["y"].shift(7).iloc[-TEST_DAYS:].values
actual = test["y"].values

baseline_mae = mean_absolute_error(actual, naive_preds)
baseline_rmse = np.sqrt(mean_squared_error(actual, naive_preds))
baseline_mape = np.mean(np.abs((actual - naive_preds) / actual)) * 100

# -----------------------------
# 5. Main model: Prophet
# -----------------------------
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model.fit(train)

future = model.make_future_dataframe(periods=TEST_DAYS)
forecast = model.predict(future)

prophet_preds = forecast.iloc[-TEST_DAYS:]["yhat"].values

prophet_mae = mean_absolute_error(actual, prophet_preds)
prophet_rmse = np.sqrt(mean_squared_error(actual, prophet_preds))
prophet_mape = np.mean(np.abs((actual - prophet_preds) / actual)) * 100

# -----------------------------
# 6. Evaluation summary
# -----------------------------
results = pd.DataFrame({
    "Model": ["Seasonal Naive (baseline)", "Prophet"],
    "MAE": [baseline_mae, prophet_mae],
    "RMSE": [baseline_rmse, prophet_rmse],
    "MAPE (%)": [baseline_mape, prophet_mape]
})
print("\n=== Model Evaluation ===")
print(results.to_string(index=False))
results.to_csv("outputs/evaluation_results.csv", index=False)

# Plot actual vs predicted
plt.figure(figsize=(12, 5))
plt.plot(test["ds"], actual, label="Actual", marker="o")
plt.plot(test["ds"], prophet_preds, label="Prophet Forecast", marker="x")
plt.plot(test["ds"], naive_preds, label="Naive Baseline", linestyle="--", alpha=0.6)
plt.legend()
plt.title("Actual vs Forecasted Sales (Last 30 Days)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=100)
print("Saved comparison plot -> outputs/actual_vs_predicted.png")

# -----------------------------
# 7. Retrain on FULL data and save model for the Streamlit app
# -----------------------------
final_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
final_model.fit(df)

with open("model.pkl", "wb") as f:
    pickle.dump(final_model, f)
print("\nSaved final trained model -> model.pkl")

print("\nDone. This model is trained on ALL available data and ready for the Streamlit dashboard.")

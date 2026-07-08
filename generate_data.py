"""
Generates a realistic synthetic daily sales dataset for one store/item.
Includes: upward trend, yearly seasonality, weekly seasonality, holiday spikes, and noise.
Run this once to create data/sales.csv
"""
import pandas as pd
import numpy as np

np.random.seed(42)

start_date = "2021-01-01"
end_date = "2023-12-31"
dates = pd.date_range(start=start_date, end=end_date, freq="D")

n = len(dates)
t = np.arange(n)

# 1. Upward trend (business growing slowly over time)
trend = 200 + 0.05 * t

# 2. Yearly seasonality (higher sales in Nov-Dec, lower in Feb)
yearly_seasonality = 40 * np.sin(2 * np.pi * t / 365.25 + 1.5)

# 3. Weekly seasonality (weekends higher than weekdays)
day_of_week = dates.dayofweek
weekly_seasonality = np.where(day_of_week >= 5, 30, 0)  # Sat/Sun boost

# 4. Random noise
noise = np.random.normal(0, 15, n)

# 5. Holiday spikes (a few fixed dates each year, e.g. festive season)
holiday_boost = np.zeros(n)
for year in range(2021, 2024):
    for md in ["12-24", "12-25", "11-25", "01-01"]:
        holiday_date = pd.Timestamp(f"{year}-{md}")
        mask = (dates >= holiday_date - pd.Timedelta(days=2)) & (dates <= holiday_date + pd.Timedelta(days=1))
        holiday_boost[mask] += 60

sales = trend + yearly_seasonality + weekly_seasonality + noise + holiday_boost
sales = np.clip(sales, a_min=20, a_max=None)  # sales can't go negative
sales = sales.round().astype(int)

df = pd.DataFrame({
    "date": dates,
    "store": 1,
    "item": 1,
    "sales": sales
})

df.to_csv("data/sales.csv", index=False)
print(f"Generated {len(df)} rows -> data/sales.csv")
print(df.head())
print(df.tail())

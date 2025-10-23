import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# data loading
url = "https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv"
df = pd.read_csv(url, parse_dates=["Date"])


# data exploration
# print(df.head())
# print(df.info())
# print(df.describe())


# plt.figure(figsize=(14, 6))

# plt.plot(df["Date"], df["Consumption"], label="Consumption")
# plt.xlabel("Date")
# plt.ylabel("Consumption")
# plt.show()

# create new time features
df["Hour"] = df["Date"].dt.hour
df["DayOfWeek"] = df["Date"].dt.day_of_week
df["Month"] = df["Date"].dt.month
df["IsWeekend"] = df["Date"].dt.day_of_week >= 5

# lag features
df["Lag_1"] = df["Consumption"].shift(1)
df["Lag_7"] = df["Consumption"].shift(7)
df["Lag_365"] = df["Consumption"].shift(365)

# rolling avg.
df["MA_7"] = df["Consumption"].rolling(window=7).mean()
df["MA_30"] = df["Consumption"].rolling(window=30).mean()


# seasons
def get_seasons(month):
    winter_months = [1, 2, 12]
    spring_months = [3, 4, 5]
    summer_months = [6, 7, 8]
    fall_months = [9, 10, 11]
    if month in winter_months:
        return "Winter"
    elif month in spring_months:
        return "Spring"
    elif month in summer_months:
        return "Summer"
    elif month in fall_months:
        return "Fall"
    else:
        return ""


df["Season"] = df["Month"].apply(get_seasons)

df = df.dropna()

split_index = int(len(df) * 0.8)
train = df[:split_index]
test = df[split_index:]

print(f"Train: {train['Date'].min()} to {train['Date'].max()}")
print(f"Test: {test['Date'].min()} to {test['Date'].max()}")

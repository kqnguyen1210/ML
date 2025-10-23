import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

# model fitting and predictions
feature_cols = ["IsWeekend", "Lag_1", "Lag_7", "Season", "MA_7", "MA_30"]

X_train = train[feature_cols]
X_test = test[feature_cols]
y_train = train["Consumption"]
y_test = test["Consumption"]

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

rf_model = RandomForestRegressor(n_estimators=500, max_depth=100, random_state=52)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

train_target = train.set_index("Date")["Consumption"]

sarima_model = SARIMAX(train_target, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
sarima_fit = sarima_model.fit()

sarima_pred = sarima_fit.forecast(steps=len(test))

sarima_mae = mean_absolute_error(y_test, sarima_pred)
sarima_rmse = mean_squared_error(y_test, sarima_pred)

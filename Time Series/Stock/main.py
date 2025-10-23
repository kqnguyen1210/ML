import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

print(df.head())
print(df.info())
print(df.describe())

plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"])
plt.title("AAPL Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.show()

df = df.reset_index()

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.day_of_week
df["Quarter"] = df["Date"].dt.quarter

df["Close_Lag1"] = df["Close"].shift(1)
df["Close_Lag7"] = df["Close"].shift(7)
df["MA_7"] = df["Close"].rolling(window=7).mean()  # 7 day moving average
df["MA_30"] = df["Close"].rolling(window=30).mean()  # 30 day moving average

df = df.dropna()

print(df.head(35))

split_index = int(len(df) * 0.8)

train = df[:split_index]
test = df[split_index:]

print(f"Train size: {len(train)}, Test size: {len(test)}")
print(f"Train period: {train['Date'].min()} to {train['Date'].max()}")
print(f"Test period: {test['Date'].min()} to {test['Date'].max()}")


feature_cols = ["Close_Lag1", "Close_Lag7", "MA_7", "MA_30", "DayOfWeek", "Month"]
X_train = train[feature_cols]
X_test = test[feature_cols]
y_train = train["Close"]
y_test = test["Close"]


model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"MAE: ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")

plt.figure(figsize=(14, 6))
plt.plot(test["Date"], y_test, label="Actual Price", color="blue")
plt.plot(test["Date"], preds, label="Predicted Price", color="red", linestyle="--")
plt.title("Stock Price: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.show()

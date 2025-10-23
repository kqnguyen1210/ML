import pandas as pd

url = "https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv"
df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")

print(df.head())
print(df.info())
print(df.describe())

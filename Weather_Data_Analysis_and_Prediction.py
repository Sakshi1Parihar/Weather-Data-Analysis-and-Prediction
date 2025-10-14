# File: Weather_Data_Analysis_and_Prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Load and Prepare Dataset
# -----------------------------
# Make sure a file named 'weather.csv' is in the same folder as this script.
# It should have at least 'date' and 'temperature' columns.

data = pd.read_csv("weather.csv", parse_dates=['date'])

# Create a new column 'day' as the number of days since the first record
data['day'] = (data['date'] - data['date'].min()).dt.days

print("✅ Dataset Loaded Successfully!")
print(data.head())

# -----------------------------
# 2. Train a Regression Model
# -----------------------------
X = data[['day']]
y = data['temperature']

model = LinearRegression()
model.fit(X, y)

print("\n✅ Model Trained Successfully!")

# -----------------------------
# 3. Predict Future Temperatures
# -----------------------------
# Predict for the next 30 days
future_days = np.arange(data['day'].max() + 1, data['day'].max() + 31).reshape(-1, 1)
future_preds = model.predict(future_days)

# Create future dates
future_dates = pd.date_range(start=data['date'].max() + pd.Timedelta(days=1), periods=30)

# Store in DataFrame
forecast = pd.DataFrame({
    'date': future_dates,
    'predicted_temperature': future_preds
})

print("\n📅 Next 5 Days Forecast:")
print(forecast.head())

# -----------------------------
# 4. Visualize Results
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(data['date'], data['temperature'], label="Historical Data", color='blue')
plt.plot(forecast['date'], forecast['predicted_temperature'], label="Predicted Data", color='red', linestyle='--')
plt.title("🌦️ Temperature Forecast for Next 30 Days")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 5. Save Forecast to CSV
# -----------------------------
forecast.to_csv("future_forecast.csv", index=False)
print("\n💾 Forecast saved to 'future_forecast.csv'")

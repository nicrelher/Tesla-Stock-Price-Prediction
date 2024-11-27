# 'pip install pandas numpy matplotlib yfinance scikit-learn tensorflow' in terminal to install required libraries

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# get historical data for Tesla (TSLA)
ticker = 'TSLA'
data = yf.download(ticker, start = '2019-01-01', end = '2023-12-01')
print(data.head()) # first 5 rows of the data

# Feature Engineering
data['Lag_1'] = data['Close'].shift(1) # Previous day's closing price
data['Target'] = data['Close'].shift(-1) # Next day's closing price
data = data.dropna() # Remove rows with NaN values

# Define features and target
X = data[['Close', 'Lag_1']] # Features: today's and yesterday's closing price
y = data['Target'] # Target: tomorrow's closing price

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae:2f}')

# Plot results
plt.figure(figsize = (12, 6))
plt.plot(y_test.values, label = "Actual Prices", color = 'blue', alpha = 0.7)
plt.plot(predictions, label = "Predicted Prices", color = 'orange', alpha = 0.7)
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()


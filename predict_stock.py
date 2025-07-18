import yfinance as yf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ticker = "AAPL"
data = yf.download(ticker, period = "5y", auto_adjust=True)

# Feature engineering
data["Return"] = data["Close"].pct_change()
data["MA10"] = data["Close"].rolling(window=10).mean()
data["MA50"] = data["Close"].rolling(window=50).mean()
data["Momentum"] = data["Close"] - data["Close"].shift(5)
data["Volatility"] = data["Return"].rolling(window=5).std()
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
data.dropna(inplace=True)

# Prepare inputs
features = ["Return", "MA10", "MA50", "Momentum", "Volatility",]
X = data[features]
Y = data["Target"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.4)

model = RandomForestClassifier(class_weight="balanced", random_state=10)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))

data_test = data.iloc[len(X_train):].copy()
data_test["Predicted"] = y_pred

print(Y.value_counts())



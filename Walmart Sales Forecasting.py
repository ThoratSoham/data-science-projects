# ============================================================
# Walmart Sales Forecasting - XGBoost Regression Model
# ============================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


# -----------------------------
# 2. Load Data
# -----------------------------
df_train = pd.read_csv(".\data\train.csv")
df_test = pd.read_csv(".\data\test.csv")
df_stores = pd.read_csv(".\data\stores.csv")
df_features = pd.read_csv(".\data\features.csv")


# -----------------------------
# 3. Data Preprocessing
# -----------------------------

# Convert Date columns to datetime
df_train["Date"] = pd.to_datetime(df_train["Date"])
df_test["Date"] = pd.to_datetime(df_test["Date"])
df_features["Date"] = pd.to_datetime(df_features["Date"])

# Convert boolean to integer
df_train.replace({False: 0, True: 1}, inplace=True)
df_test.replace({False: 0, True: 1}, inplace=True)
df_features.replace({False: 0, True: 1}, inplace=True)

# Fill missing markdown values with 0
markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
df_features[markdown_cols] = df_features[markdown_cols].fillna(0)

# Fill economic indicators with median
df_features["CPI"].fillna(df_features["CPI"].median(), inplace=True)
df_features["Unemployment"].fillna(df_features["Unemployment"].median(), inplace=True)

# Merge datasets
df_train = df_train.merge(df_stores, on="Store", how="left")
df_test = df_test.merge(df_stores, on="Store", how="left")

df_train = df_train.merge(df_features, on=["Store", "Date"], how="left")
df_test = df_test.merge(df_features, on=["Store", "Date"], how="left")


# -----------------------------
# 4. Feature Engineering
# -----------------------------

# Extract date features
for df in [df_train, df_test]:
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

# Drop original Date column
df_train.drop("Date", axis=1, inplace=True)
df_test.drop("Date", axis=1, inplace=True)


# -----------------------------
# 5. Train / Validation Split
# -----------------------------
X = df_train.drop("Weekly_Sales", axis=1)
y = df_train["Weekly_Sales"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=70
)


# -----------------------------
# 6. Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# -----------------------------
# 7. Model Training (XGBoost)
# -----------------------------
model = XGBRegressor(
    n_estimators=900,
    learning_rate=0.01,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=70
)

model.fit(X_train_scaled, y_train)


# -----------------------------
# 8. Model Evaluation
# -----------------------------
predictions = model.predict(X_val_scaled)

rmse = np.sqrt(mean_squared_error(y_val, predictions))
mae = mean_absolute_error(y_val, predictions)
r2 = r2_score(y_val, predictions)

print("Model Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.4f}")


# -----------------------------
# 9. Visualization
# -----------------------------

# Actual vs Predicted Plot
plt.figure(figsize=(8,6))
plt.scatter(y_val, predictions, alpha=0.5)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
plt.barh(importance_df["Feature"][:15], importance_df["Importance"][:15])
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances")
plt.show()

# =========================================================
# 1. IMPORTS
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv("data/Telco_customer_churn.csv")

# =========================================================
# 3. BASIC CLEANING
# =========================================================

df['Total Charges'] = pd.to_numeric(df['Total Charges'].str.strip(), errors='coerce')
df['Total Charges'].fillna(df['Total Charges'].mean(), inplace=True)

df = df.drop([
    'Unnamed: 0', 'CustomerID', 'Count', 'Country',
    'State', 'Churn Reason', 'Lat Long', 'Zip Code',
    'Latitude', 'Longitude'
], axis=1)


# =========================================================
# 4. FEATURE / TARGET SPLIT
# =========================================================

X = df.drop(['Churn Value', 'Churn Label'], axis=1)
y = df['Churn Value']


# Convert binary columns
binary_cols = [
    'Phone Service', 'Multiple Lines', 'Online Security',
    'Online Backup', 'Device Protection', 'Tech Support',
    'Streaming TV', 'Streaming Movies', 'Paperless Billing',
    'Partner', 'Dependents', 'Senior Citizen'
]

for col in binary_cols:
    X[col] = X[col].replace({
        'Yes': 1,
        'No': 0,
        'No phone service': 0,
        'No internet service': 0
    })

X['Internet Service'] = X['Internet Service'].replace({0: 'No'})


# =========================================================
# 5. TRAIN TEST SPLIT
# =========================================================

numeric_features = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV'] + binary_cols
categorical_features = ['City', 'Gender', 'Internet Service', 'Contract', 'Payment Method']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=70, stratify=y
)


# =========================================================
# 6. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

cv_scores = cross_val_score(
    xgb_model,
    X_train_transformed,
    y_train,
    cv=5,
    scoring="roc_auc"
)

print("Cross-Validated ROC-AUC:", cv_scores.mean())

# =========================================================
# 7. HANDLE CLASS IMBALANCE (SMOTE)
# =========================================================

smote = SMOTE(random_state=70)
X_train_bal, y_train_bal = smote.fit_resample(X_train_transformed, y_train)


# =========================================================
# 8. BASELINE MODEL — XGBOOST
# =========================================================

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    random_state=70
)

xgb_model.fit(X_train_bal, y_train_bal)

y_pred = xgb_model.predict(X_test_transformed)
y_prob = xgb_model.predict_proba(X_test_transformed)[:, 1]

print("XGBoost Results")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# =========================================================
# 9. RANDOM FOREST MODEL
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    class_weight="balanced",
    random_state=70
)

rf_model.fit(X_train_bal, y_train_bal)

y_pred_rf = rf_model.predict(X_test_transformed)
y_prob_rf = rf_model.predict_proba(X_test_transformed)[:, 1]

print("\nRandom Forest Results")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))


# =========================================================
# 10. HYPERPARAMETER TUNING (XGBOOST)
# =========================================================

param_dist = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1],
    "colsample_bytree": [0.6, 0.8, 1]
}

random_search = RandomizedSearchCV(
    XGBClassifier(objective="binary:logistic", eval_metric="auc", random_state=70),
    param_distributions=param_dist,
    n_iter=20,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train_bal, y_train_bal)

best_model = random_search.best_estimator_

y_pred_best = best_model.predict(X_test_transformed)
y_prob_best = best_model.predict_proba(X_test_transformed)[:, 1]

print("\nBest XGBoost Results")
print(classification_report(y_test, y_pred_best))
print("Best ROC-AUC:", roc_auc_score(y_test, y_prob_best))

# =========================================================
# 11. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_test, y_pred_best)

plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - Best XGBoost")
plt.colorbar()
plt.xticks([0,1], ["No Churn","Churn"])
plt.yticks([0,1], ["No Churn","Churn"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

# =========================================================
# 11. ROC CURVE
# =========================================================

fpr, tpr, _ = roc_curve(y_test, y_prob_best)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Best XGBoost")
plt.show()


# =========================================================
# 12. THRESHOLD TUNING
# =========================================================

print("\nF1 Scores at Different Thresholds")
thresholds = np.arange(0.1, 0.9, 0.05)

for t in thresholds:
    y_pred_custom = (y_prob_best > t).astype(int)
    print(f"Threshold {t:.2f} F1:", f1_score(y_test, y_pred_custom))


# =========================================================
# 13. EXPLAINABILITY (SHAP)
# =========================================================

explainer = shap.Explainer(best_model)
shap_values = explainer(X_test_transformed)

shap.summary_plot(shap_values, X_test_transformed)

feature_names = preprocessor.get_feature_names_out()
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)

shap.plots.waterfall(shap_values[0])

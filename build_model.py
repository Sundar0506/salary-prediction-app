"""
build_model.py
--------------
Train Employee Salary Prediction models (Linear Regression & Random Forest)
from the Edunet Foundation salary dataset. Evaluate and save the best model
(Random Forest) as salary_prediction_model.pkl (joblib format).
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ------------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------------
HERE = Path(__file__).parent.resolve()
DATA_PATH = HERE / "Salary Data.csv"   # make sure the CSV is in the same folder
MODEL_PATH = HERE / "salary_prediction_model.pkl"

# ------------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("\n=== Loaded Data (first 5 rows) ===")
print(df.head())
print("\nShape before cleaning:", df.shape)
print("\nMissing values:\n", df.isna().sum())

# ------------------------------------------------------------------
# 3. Drop rows with missing target (Salary)
# ------------------------------------------------------------------
df = df.dropna(subset=['Salary'])
print("\nShape after dropping rows with missing Salary:", df.shape)

# ------------------------------------------------------------------
# 4. Feature / target setup
# ------------------------------------------------------------------
TARGET_COL = 'Salary'
categorical_features = ['Gender', 'Education Level', 'Job Title']
numeric_features = ['Age', 'Years of Experience']

X = df[categorical_features + numeric_features]
y = df[TARGET_COL]

# ------------------------------------------------------------------
# 5. Preprocessing pipelines
# ------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ]
)

# ------------------------------------------------------------------
# 6. Train / Test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)

# ------------------------------------------------------------------
# 7. Linear Regression
# ------------------------------------------------------------------
lr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\n--- Linear Regression Performance ---")
print("MAE :", mean_absolute_error(y_test, y_pred_lr))
print("MSE :", mean_squared_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2  :", r2_score(y_test, y_pred_lr))

# ------------------------------------------------------------------
# 8. Random Forest
# ------------------------------------------------------------------
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest Performance ---")
print("MAE :", mean_absolute_error(y_test, y_pred_rf))
print("MSE :", mean_squared_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2  :", r2_score(y_test, y_pred_rf))

# ------------------------------------------------------------------
# 9. Save best model (Random Forest)
# ------------------------------------------------------------------
joblib.dump(rf_model, MODEL_PATH)
print(f"\n✅ Model saved as {MODEL_PATH}")

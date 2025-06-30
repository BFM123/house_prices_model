"""
House Price Prediction Script
=============================

This script trains a linear regression model to predict house prices
and generates a submission CSV.

USAGE:
------

Default (if your data is in 'data/train.csv' and 'data/test.csv'):
    python train_and_predict.py

Specify custom file paths:
    python train_and_predict.py --train_path my_train.csv --test_path my_test.csv --output_path my_submission.csv

Dependencies:
    pip install -r requirements.txt
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# --------------------------
# CLI ARGUMENTS
# --------------------------
parser = argparse.ArgumentParser(description="Train linear regression model and predict house prices.")
parser.add_argument("--train_path", type=str, default="data/train.csv", help="Path to training CSV.")
parser.add_argument("--test_path", type=str, default="data/test.csv", help="Path to test CSV.")
parser.add_argument("--output_path", type=str, default="submission.csv", help="Output CSV file for predictions.")

args = parser.parse_args()

train_path = args.train_path
test_path = args.test_path
output_path = args.output_path

print(f"\n✅ Loading data:\n   Train: {train_path}\n   Test:  {test_path}")

# --------------------------
# LOAD DATA
# --------------------------
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# --------------------------
# FEATURES
# --------------------------
numeric_features = [
    'GrLivArea', 'OverallQual', 'TotalBsmtSF', 'GarageCars',
    'YearBuilt', '1stFlrSF', 'Fireplaces'
]
categorical_features = ['Neighborhood']

# One-hot encode
train_df = pd.get_dummies(train_df, columns=categorical_features, drop_first=True)
test_df = pd.get_dummies(test_df, columns=categorical_features, drop_first=True)

# Align columns
missing_cols = set(train_df.columns) - set(test_df.columns)
for col in missing_cols:
    if col.startswith("Neighborhood_"):
        test_df[col] = 0

test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

# --------------------------
# BUILD X AND y
# --------------------------
X = train_df[numeric_features + [c for c in train_df.columns if c.startswith("Neighborhood_")]]
X_test = test_df[numeric_features + [c for c in train_df.columns if c.startswith("Neighborhood_")]]
y = np.log1p(train_df["SalePrice"])

# --------------------------
# PIPELINE
# --------------------------
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# --------------------------
# TRAIN
# --------------------------
print("\n✅ Training model...")
pipeline.fit(X, y)

# --------------------------
# EVALUATE
# --------------------------
print("\n✅ Evaluating with 5-fold cross-validation...")
cv_rmse = -cross_val_score(
    pipeline, X, y,
    scoring="neg_root_mean_squared_error",
    cv=5
).mean()
print(f"   Cross-validated RMSE (log scale): {cv_rmse:.4f}")

# --------------------------
# PREDICT
# --------------------------
print("\n✅ Predicting on test data...")
log_preds = pipeline.predict(X_test)
preds = np.expm1(log_preds)

# --------------------------
# SAVE
# --------------------------
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": preds
})
submission.to_csv(output_path, index=False)
print(f"\n✅ Predictions saved to: {output_path}\n")
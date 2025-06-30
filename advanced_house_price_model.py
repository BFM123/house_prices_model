import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Features
numeric_features = [
    'LotArea', 'GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath',
    'OverallQual', 'OverallCond', 'TotalBsmtSF', 'GarageCars', 'GarageArea',
    'YearBuilt', 'YearRemodAdd'
]
categorical_features = ['Neighborhood']

# One-hot encode categoricals
train_df = pd.get_dummies(train_df, columns=categorical_features, drop_first=True)
test_df = pd.get_dummies(test_df, columns=categorical_features, drop_first=True)

# Align columns
missing_cols_in_test = set(train_df.columns) - set(test_df.columns)
for col in missing_cols_in_test:
    if col.startswith("Neighborhood_"):
        test_df[col] = 0
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

# Target log-transform
y = np.log1p(train_df['SalePrice'])

# Feature matrix
X = train_df[numeric_features + [c for c in train_df.columns if c.startswith("Neighborhood_")]]
X_test = test_df[numeric_features + [c for c in test_df.columns if c.startswith("Neighborhood_")]]

# Fill missing
X.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

# Scale numeric
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Model
model = LinearRegression()
model.fit(X, y)

# Cross-validation
cv_rmse = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=5).mean()
print(f"Linear Regression CV RMSE (log-transformed target): {cv_rmse:.4f}")

# Predict
log_preds = model.predict(X_test)
preds = np.expm1(log_preds)  # Convert back to original scale

# Save submission
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": preds
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created!")
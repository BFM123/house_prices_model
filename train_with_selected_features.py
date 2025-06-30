import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Define selected features (without SalePrice!)
selected_features = ['GrLivArea', 'OverallQual', 'TotalBsmtSF', 'GarageCars', 'YearBuilt', '1stFlrSF', 'Fireplaces']

# One-hot encode categoricals
categorical_features = ['Neighborhood']
train_df = pd.get_dummies(train_df, columns=categorical_features, drop_first=True)
test_df = pd.get_dummies(test_df, columns=categorical_features, drop_first=True)

# Align columns
missing_cols = set(train_df.columns) - set(test_df.columns)
for col in missing_cols:
    if col.startswith("Neighborhood_"):
        test_df[col] = 0
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

# Build X and y
X = train_df[selected_features + [c for c in train_df.columns if c.startswith("Neighborhood_")]]
X_test = test_df[selected_features + [c for c in train_df.columns if c.startswith("Neighborhood_")]]
y = np.log1p(train_df['SalePrice'])

# Fill missing
X = X.fillna(0)
X_test = X_test.fillna(0)

# Standardize numeric
scaler = StandardScaler()
X.loc[:, selected_features] = scaler.fit_transform(X[selected_features])
X_test.loc[:, selected_features] = scaler.transform(X_test[selected_features])

# Train model
model = LinearRegression()
model.fit(X, y)

# Evaluate CV RMSE
cv_rmse = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=5).mean()
print(f"Linear Regression CV RMSE with selected features: {cv_rmse:.4f}")

# Predict
log_preds = model.predict(X_test)
preds = np.expm1(log_preds)

# Save submissions
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": preds
})
submission.to_csv("submission.csv", index=False)
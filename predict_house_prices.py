import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load training and test data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Define features
features = [
    'LotArea',
    'GrLivArea',
    'BedroomAbvGr',
    'FullBath',
    'HalfBath',
    'OverallQual',
    'OverallCond',
    'TotalBsmtSF',
    'GarageCars',
    'GarageArea',
    'YearBuilt',
    'YearRemodAdd',
    'Fireplaces'
]

# One-hot encode 'Neighborhood' in both train and test
train_df = pd.get_dummies(train_df, columns=['Neighborhood'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Neighborhood'], drop_first=True)

# Align test and train columns (make sure test has same dummies)
train_columns = set(train_df.columns)
test_columns = set(test_df.columns)
missing_cols_in_test = train_columns - test_columns
for col in missing_cols_in_test:
    if col.startswith("Neighborhood_"):
        test_df[col] = 0
# Make sure columns are in the same order
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

# Target
y = train_df['SalePrice']

# Features + Neighborhood dummies
X_train = train_df[features + [col for col in train_df.columns if col.startswith('Neighborhood_')]]
X_test = test_df[features + [col for col in train_df.columns if col.startswith('Neighborhood_')]]

# Fill missing numeric values (if any)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

# Standardize numeric features
scaler = StandardScaler()
X_train[features] = scaler.fit_transform(X_train[features])
X_test[features] = scaler.transform(X_test[features])

# Train the model
model = LinearRegression()
model.fit(X_train, y)

# Predict on test data
test_preds = model.predict(X_test)

# Prepare submission dataframe
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_preds
})

# Save submission
submission.to_csv("submission.csv", index=False)

print("✅ submission.csv created!")
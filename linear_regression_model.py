import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import time

# Loading the dataset
df = pd.read_csv("data/train.csv")

# Selecting the relevant features
features = [
    'LotArea',         # Lot square footage
    'GrLivArea',       # Above ground living area square footage
    'BedroomAbvGr',    # Number of bedrooms above ground
    'FullBath',        # Number of full bathrooms
    'HalfBath',        # Number of half bathrooms
    'OverallQual',
    'OverallCond',
    'TotalBsmtSF',
    'GarageCars',
    'GarageArea',
    'YearBuilt',
    'YearRemodAdd',
    'Fireplaces'
]
target = 'SalePrice'

# One-hot encode Neighborhood efficiently
df = pd.get_dummies(df, columns=['Neighborhood'], drop_first=True)

# Define features and target
X = df[features + [col for col in df.columns if col.startswith('Neighborhood_')]]
y = df[target]

# Scale numerical features (only those that are numeric and not one-hot columns)
scaler = StandardScaler()
X[features] = scaler.fit_transform(X[features])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and time the model
start = time.time()
model = LinearRegression()
model.fit(X_train, y_train)
end = time.time()

# Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = model.score(X_test, y_test)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2 Score: {r2:.4f}")
print(f"Training Time: {end - start:.4f} seconds")

# Show a few predictions vs actual values for inspection
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
print("\nSample predictions:")
print(results.head(10))
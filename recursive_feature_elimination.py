from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Load data
train_df = pd.read_csv("data/train.csv")

# Define numeric features
numeric_features = [
    'LotArea', 'GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath',
    'OverallQual', 'OverallCond', 'TotalBsmtSF', 'GarageCars', 'GarageArea',
    'YearBuilt', 'YearRemodAdd'
]

# Create X and y
X = train_df[numeric_features].fillna(0)
y = np.log1p(train_df['SalePrice'])

# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit RFE
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=8, step=1)
selector = selector.fit(X_scaled, y)

# Print selected features
selected = np.array(numeric_features)[selector.support_]
print("Selected features:", selected)
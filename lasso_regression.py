from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load data
train_df = pd.read_csv("data/train.csv")

# Define numeric features
numeric_features = [
    'LotArea', 'GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath',
    'OverallQual', 'OverallCond', 'TotalBsmtSF', 'GarageCars', 'GarageArea',
    'YearBuilt', 'YearRemodAdd'
]

X = train_df[numeric_features].fillna(0)
y = np.log1p(train_df['SalePrice'])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LassoCV(cv=5, random_state=0)
lasso.fit(X_scaled, y)

# Show coefficients
coef_df = pd.Series(lasso.coef_, index=numeric_features)
print(coef_df.sort_values())
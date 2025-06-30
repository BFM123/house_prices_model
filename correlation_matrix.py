import pandas as pd

train_df = pd.read_csv("data/train.csv")

numeric_features = [
    'LotArea', 'GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath',
    'OverallQual', 'OverallCond', 'TotalBsmtSF', 'GarageCars',
    'GarageArea', 'YearBuilt', 'YearRemodAdd',
    '1stFlrSF', '2ndFlrSF', 'WoodDeckSF', 'OpenPorchSF',
    'EnclosedPorch', 'TotRmsAbvGrd', 'Fireplaces'
]

corr = train_df[numeric_features + ['SalePrice']].corr()
print(corr['SalePrice'].sort_values(ascending=False))
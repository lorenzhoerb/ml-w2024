import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor as RandomForestRegressor_sklearn
from sklearn.model_selection import train_test_split, cross_validate

# from sklearn.metrics import root_mean_squared_error

from random_forest.random_forest import RandomForestRegressor

data = pd.read_csv("data/bike_sharing/day.csv")

# split
X = data.iloc[:, 2:-3].values # exclude instant, casual, and registered
Y = data.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# train
regressor_our = RandomForestRegressor(tree_min_nodes=2, tree_max_depth=10)
# regressor.fit(X_train, Y_train)
regressor_sklearn = RandomForestRegressor_sklearn() 

regressor_our.fit(X_train, Y_train)

Y_pred = regressor_our.predict(X_test)

print(Y_pred)

# asd = cross_validate(regressor_our, X, Y, scoring='neg_root_mean_squared_error', cv=5)

# print(asd)


# predict
# Y_pred = regressor.predict(X_test)
# print(Y_pred)

# def asd(X, Y, cv):
#     num_samples = X.shape[0]
#     for i in range(cv):
#         _X = X[]

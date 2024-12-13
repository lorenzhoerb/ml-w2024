import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from random_forest.decision_tree import DecisionTree
from sklearn.tree import DecisionTreeRegressor as RegressionTree_sklearn
from random_forest.regression_tree import RegressionTree

data = pd.read_csv("data/bike_sharing/day.csv")

# split
X = data.iloc[:, 2:-3].values # exclude instant, casual, and registered
Y = data.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# train - sklearn
# regressor = DecisionTree(min_samples_split=20, max_depth=10)
regressor = RegressionTree_sklearn(min_samples_split=20, max_depth=10)
regressor.fit(X_train, Y_train)

# predict - sklearn
Y_pred = regressor.predict(X_test)
print("Mean squared error: ", mean_squared_error(Y_test, Y_pred))
print("Actual | Predicted")
print("--------------------")
i = 0
while i < 10:
    print(Y_test[i], "|", Y_pred[i])
    i += 1

print("---------- Custom Solution ----------")

# train
regressor = RegressionTree(min_nodes=20, max_depth=10)
regressor.fit(X_train, Y_train)
# regressor.print_tree()

# predict
Y_pred = regressor.predict(X_test)
print("Mean squared error: ", mean_squared_error(Y_test, Y_pred))
print("Actual | Predicted")
print("--------------------")

for i in range(Y_test.shape[0])[:10]:
    print(Y_test[i], "|", Y_pred[i])
    #   if Y_pred[i] is None or math.isnan(Y_pred[i]):
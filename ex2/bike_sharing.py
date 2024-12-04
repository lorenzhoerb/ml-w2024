import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from random_forest.decision_tree import DecisionTree

data = pd.read_csv("data/bike_sharing/day.csv")

# split
X = data.iloc[:, 1:-3].values # exclude instant, casual, and registered
Y = data.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=0)

# train
regressor = DecisionTree(min_samples_split=20, max_depth=10)
regressor.fit(X_train,Y_train)
# regressor.print_tree()

# predict
Y_pred = regressor.predict(X_test)
print("Mean squared error: ", mean_squared_error(Y_test, Y_pred))
print("Actual | Predicted")
print("--------------------")
i = 0
while i < 10:
    print(Y_test[i], "|", Y_pred[i])
    i += 1
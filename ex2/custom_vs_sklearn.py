import pandas as pd
from sklearn.metrics import mean_squared_error, \
    root_mean_squared_error as RMSE, \
    median_absolute_error as MAE, \
    r2_score as R2
from sklearn.ensemble import RandomForestRegressor as RandomForestRegressor_sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_validate
import time 
import json
from random_forest.random_forest import RandomForestRegressor

def prepare_datasets():
  datasets = {}
  data = pd.read_csv("data/bike_sharing/day.csv")

  # split
  X = data.iloc[:, 2:-3].values # exclude instant, casual, and registered
  Y = data.iloc[:, -1].values.reshape(-1,1)

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
  datasets["bike"] = {
      "X_train": X_train,
      "Y_train": Y_train,
      "X_test": X_test,
      "Y_test": Y_test
  }

  data = pd.read_csv("data/online_news_popularity/OnlineNewsPopularity/OnlineNewsPopularity.csv")
  X = data.iloc[:200, 1:-1].values
  Y = data.iloc[:200, -1].values.reshape(-1,1)

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
  datasets["popularity"] = {
      "X_train": X_train,
      "Y_train": Y_train,
      "X_test": X_test,
      "Y_test": Y_test
  }

  return datasets


def compare_regressors(regressor1, regressor2, dataset):
    scores = (
        get_scores_for_regressor(regressor1, dataset["X_train"], dataset["X_test"], dataset["Y_train"], dataset["Y_test"]),
        get_scores_for_regressor(regressor2, dataset["X_train"], dataset["X_test"], dataset["Y_train"], dataset["Y_test"]),
    )

    return {
        "RMSE": (scores[0]["RMSE"], scores[1]["RMSE"]),
        "MAE": (scores[0]["MAE"], scores[1]["MAE"]),
        "R2": (scores[0]["R2"], scores[1]["R2"]),
        "Runtime (s)": (scores[0]["Runtime (s)"], scores[1]["Runtime (s)"]),
    }


def get_scores_for_regressor(regressor, _X_train, _X_test, _Y_train, _Y_test) -> dict:
    start = time.time()
    regressor.fit(_X_train, _Y_train)
    Y_pred = regressor.predict(_X_test)
    runtime = time.time() - start

    return {
        "RMSE": RMSE(_Y_test, Y_pred),
        "MAE": MAE(_Y_test, Y_pred),
        "R2": R2(_Y_test, Y_pred),
        "Runtime (s)": runtime
    }


datasets = prepare_datasets()

# Custom RandomForestRegressor, vs sklearn RandomForestRegressor
comparison = [
    (RandomForestRegressor(n_trees=100, tree_max_depth=10, tree_min_nodes=2), RandomForestRegressor_sklearn()),
    (RandomForestRegressor(n_trees=50, tree_max_depth=5, tree_min_nodes=2), RandomForestRegressor_sklearn()),
    (RandomForestRegressor(n_trees=50, tree_max_depth=20, tree_min_nodes=6), RandomForestRegressor_sklearn()),
    (RandomForestRegressor(n_trees=200, tree_max_depth=3, tree_min_nodes=10), RandomForestRegressor_sklearn()),
    (RandomForestRegressor(n_trees=200, tree_max_depth=30, tree_min_nodes=3), RandomForestRegressor_sklearn()),
]

results = [
    compare_regressors(regressor_our, regressor_sklearn, datasets["popularity"])
    for regressor_our, regressor_sklearn in comparison
]
print('Custom RandomForestRegressor, vs sklearn RandomForestRegressor')
print('---------------------------------------------------------')
print(json.dumps(results, indent=" "))

# Custom RandomForestRegressor, vs sklearn KNN
comparison_kn = [
    (RandomForestRegressor(n_trees=100, tree_max_depth=10, tree_min_nodes=2), KNeighborsRegressor(n_neighbors=5, weights='distance')),
    (RandomForestRegressor(n_trees=50, tree_max_depth=5, tree_min_nodes=2), KNeighborsRegressor(n_neighbors=5, weights='distance')),
    (RandomForestRegressor(n_trees=50, tree_max_depth=20, tree_min_nodes=6), KNeighborsRegressor(n_neighbors=5, weights='distance')),
    (RandomForestRegressor(n_trees=200, tree_max_depth=3, tree_min_nodes=10), KNeighborsRegressor(n_neighbors=5, weights='distance')),
    (RandomForestRegressor(n_trees=200, tree_max_depth=30, tree_min_nodes=3), KNeighborsRegressor(n_neighbors=5, weights='distance')),
]

results_kn = [
    compare_regressors(regressor_our, regressor_kn, datasets["popularity"])
    for regressor_our, regressor_kn in comparison_kn
]
print('Custom RandomForestRegressor, vs sklearn KNeighborsRegressor')
print('---------------------------------------------------------')
print(json.dumps(results_kn, indent=" "))
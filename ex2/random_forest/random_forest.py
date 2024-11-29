import pandas as pd
from typing import List

from .decision_tree import DecisionTree


class RandomForest:
    def __init__(self, max_depth: int = None, n_trees: int = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees: List[DecisionTree] = []

    def _bootstrap(self, X: pd.DataFrame, y: pd.Series) -> (pd.DataFrame, pd.DataFrame):
        pass

    def _fit_tree(self, X: pd.DataFrame, y: pd.Series) -> DecisionTree:
        # Create a decision tree and train it
        tree = DecisionTree(max_depth=self.max_depth)
        tree.fit(X, y)
        return tree

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap(X, y)
            tree = self._fit_tree(X_sample, y_sample)
            self.trees.append(tree)

    def _majority_prediction(self, x: pd.DataFrame) -> pd.Series:
        ## get the aggregate results
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self._majority_prediction(X)


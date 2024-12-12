from typing import List
import numpy as np
import random

from .regression_tree import RegressionTree


class RandomForest:
    def __init__(self, max_depth: int = None, n_trees: int = 100, random_state: int = 0):
        self.random_state = random_state
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees: List[RegressionTree] = []

    def _bootstrap(self, X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Take random slice of samples and features
        """
        n_samples, n_features = X.shape
        random.seed(self.random_state)
        samples_start_i = random.randint(0, n_samples - 1)
        features_start_i = random.randint(0, n_features - 1)
        samples_end_i = random.randint(samples_start_i, n_samples - 1)
        features_end_i = random.randint(features_start_i, n_features - 1)
        return X[[samples_start_i, samples_end_i],[features_start_i, features_end_i]], y[samples_start_i, samples_end_i]

    def _fit_tree(self, X: np.ndarray, y: np.ndarray) -> RegressionTree:
        # Create a regression tree and train it
        tree = RegressionTree(max_depth=self.max_depth)
        tree.fit(X, y)
        return tree

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap(X, y)
            tree = self._fit_tree(X_sample, y_sample)
            self.trees.append(tree)

    def _majority_prediction(self, x: np.ndarray) -> np.ndarray:
        ## get the aggregate results
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._majority_prediction(X)


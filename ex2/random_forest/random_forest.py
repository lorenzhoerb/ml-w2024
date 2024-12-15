from typing import List
import numpy as np

from .regression_tree import RegressionTree


class RandomForestRegressor:
    DEFAULT_TREE_MAX_DEPTH = RegressionTree.DEFAULT_MAX_DEPTH
    DEFAULT_N_TREES = 100
    DEFAULT_TREE_MIN_NODES = RegressionTree.DEFAULT_MIN_NODES
    DEFAULT_RANDOM_STATE = 0

    def __init__(self, 
                 tree_max_depth: int = DEFAULT_TREE_MAX_DEPTH, 
                 n_trees: int = DEFAULT_N_TREES, 
                 tree_min_nodes: int = DEFAULT_TREE_MIN_NODES, 
                 random_state: int = DEFAULT_RANDOM_STATE):
        self.random_state = random_state
        self.n_trees = n_trees
        self.tree_max_depth = tree_max_depth
        self.tree_min_nodes = tree_min_nodes
        self.trees: List[RegressionTree] = []

    def _bootstrap(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Create random subsample of X and y with replacement that's ~60% of the original size
        """
        np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        subsample_index = np.random.choice(n_samples, n_samples, replace=True)
        
        return X[subsample_index], y[subsample_index]

    def _fit_tree(self, X: np.ndarray, y: np.ndarray) -> RegressionTree:
        # Create a regression tree and train it
        tree = RegressionTree(min_nodes=self.tree_min_nodes, max_depth=self.tree_max_depth)
        tree.fit(X, y)
        return tree

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap(X, y)
            tree = self._fit_tree(X_sample, y_sample)
            self.trees.append(tree)

    def _majority_prediction(self, x: np.ndarray) -> np.ndarray:
        # print([tree.predict(x) for tree in self.trees])
        return np.mean([tree.predict(x) for tree in self.trees], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._majority_prediction(X)


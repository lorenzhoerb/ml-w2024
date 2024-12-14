"""
ChatGPT 4o mini
Prompt: Write a random forest regression implementation in Python from scratch without using sklearn
"""
import numpy as np
from random_forest_ChatGPT.regression_tree import DecisionTreeRegressor

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
    
    def fit(self, X, y):
        self.trees = []
        m, n = X.shape
        for _ in range(self.n_estimators):
            # Bootstrap sample of data
            indices = np.random.choice(m, m, replace=True)
            X_bootstrap, y_bootstrap = X[indices], y[indices]
            
            # Random feature subset
            if self.max_features is None:
                max_features = n
            else:
                max_features = self.max_features
            
            feature_indices = np.random.choice(n, max_features, replace=False)
            X_bootstrap = X_bootstrap[:, feature_indices]
            
            # Build a tree
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append((tree, feature_indices))
    
    def predict(self, X):
        # Gather predictions from all trees
        tree_preds = np.zeros((X.shape[0], self.n_estimators))
        
        for i, (tree, feature_indices) in enumerate(self.trees):
            X_sub = X[:, feature_indices]
            tree_preds[:, i] = tree.predict(X_sub)
        
        # Average predictions from all trees
        return np.mean(tree_preds, axis=1)

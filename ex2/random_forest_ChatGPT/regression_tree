"""
ChatGPT 4o mini
Prompt: Write a random forest regression implementation in Python from scratch without using sklearn
"""
import numpy as np

# Helper function to calculate the Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Helper function to calculate the best split for a node
def best_split(X, y):
    m, n = X.shape
    best_mse = float('inf')
    best_split = None
    
    # Loop through all features
    for feature_index in range(n):
        # Get all unique values of the feature
        unique_values = np.unique(X[:, feature_index])
        
        # Try splitting at every point between two consecutive values
        for value in unique_values:
            left_mask = X[:, feature_index] <= value
            right_mask = ~left_mask
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            y_left, y_right = y[left_mask], y[right_mask]
            mse = np.mean((y_left - np.mean(y_left)) ** 2) * len(y_left) + np.mean((y_right - np.mean(y_right)) ** 2) * len(y_right)
            mse /= len(y)
            
            if mse < best_mse:
                best_mse = mse
                best_split = (feature_index, value, left_mask)
    
    return best_split

# Recursive function to build the decision tree
class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
    
    def fit(self, X, y, depth=0):
        if len(np.unique(y)) == 1:
            self.value = np.mean(y)
            return
        
        if self.max_depth is not None and depth >= self.max_depth:
            self.value = np.mean(y)
            return
        
        split = best_split(X, y)
        
        if split is None:
            self.value = np.mean(y)
            return
        
        feature_index, value, left_mask = split
        right_mask = ~left_mask
        
        left_tree = DecisionTreeRegressor(self.max_depth)
        right_tree = DecisionTreeRegressor(self.max_depth)
        
        left_tree.fit(X[left_mask], y[left_mask], depth + 1)
        right_tree.fit(X[right_mask], y[right_mask], depth + 1)
        
        self.feature_index = feature_index
        self.value = None
        self.value_left = left_tree
        self.value_right = right_tree
        self.split_value = value
    
    def predict(self, X):
        if self.value is not None:
            return np.full(X.shape[0], self.value)
        
        left_mask = X[:, self.feature_index] <= self.split_value
        right_mask = ~left_mask
        
        predictions = np.zeros(X.shape[0])
        predictions[left_mask] = self.value_left.predict(X[left_mask])
        predictions[right_mask] = self.value_right.predict(X[right_mask])
        
        return predictions

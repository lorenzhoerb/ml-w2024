import pandas as pd
import numpy as np

from ex2.random_forest.loss import LossFunction, RSSLoss


class Node:
    def __init__(self, threshold: float = None, value: float = None, left: 'Node' = None, right: 'Node' = None,
                 feature_index: int = -1):  # No default value for feature_index
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.feature_index = feature_index


class RegressionTree:
    def __init__(self, loss_function: LossFunction = None, min_nodes: int = 2, max_depth: int = 10):
        if loss_function is None:
            loss_function = RSSLoss()  # Default loss function
        self.loss_function = loss_function
        self.min_nodes = min_nodes
        self.max_depth = max_depth
        self._is_fitted = False
        self._root = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._validate_inputs(X, y)
        root = self._create_tree(X.to_numpy(), y.to_numpy(), 0)
        self._root = root
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._is_fitted is False:
            raise Exception("Tree is not fitted yet.")
        self._validate_X(X)
        # TODO: implement predict
        return None

    def _create_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Creates the decision tree recursively. Stop if split criteria is reached. depth >= max_depth or min_nodes is not reached."""
        # shape of X values
        num_samples, num_features = X.shape

        # Check if further splitting is possible based on the stopping criteria (depth or minimum samples)
        if not self._can_split(num_samples, depth):
            # If not, calculate the average value of y and create a leaf node with this value
            average_value = np.mean(y)
            return Node(value=float(average_value))

        # If splitting is still possible, find best split feature

        # For each feature calculate the best split.
        best_feature_losses = [self._find_best_split(x_feature, y) for x_feature in X.T]

        # Convert to a numpy array to use argmin
        best_feature_losses = np.array(best_feature_losses)

        # Get the best feature index based on the feature with the lowest loss
        best_split_feature_index = np.argmin(best_feature_losses[:, 1])
        # Get best split values for feature feature_index
        threshold, loss = best_feature_losses[best_split_feature_index]

        # Split data at threshold
        x_left, y_left, x_right, y_right = self._split(X, y, int(best_split_feature_index), threshold)

        left_tree = self._create_tree(x_left, y_left, depth + 1)
        right_tree = self._create_tree(x_right, y_right, depth + 1)
        return Node(threshold=threshold, left=left_tree, right=right_tree, feature_index=int(best_split_feature_index))

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """
        Find the best threshold to split a feature based on the given loss function.

        Args:
                X (np.ndarray): A 1D array containing feature values.
                y (np.ndarray): A 1D array containing the target values.

        Returns:
            tuple[float, float]: A tuple containing the best threshold and the associated loss value.
                                 The threshold is the value that minimizes the loss for the given feature.
                                 The loss is the value of the loss function at that threshold.
        """
        # Generate pairs of adjacent elements
        best_split = None
        min_loss = float('inf')  # Initialize the minimum RSS with a very large number

        for i in range(len(X) - 1):
            threshold = X[i:i + 2].mean()  # Calculate the threshold as the mean of x1 and x2

            loss = self.loss_function.calculate(X, y, threshold)  # Calculate loss for the threshold

            # If this RSS is smaller than the current minimum, update the best_split
            if loss < min_loss:
                min_loss = loss
                best_split = (threshold, loss)

        return best_split

    def _can_split(self, num_samples: int, depth: int) -> bool:
        """checks if split criteria is reached."""
        return num_samples >= self.min_nodes and depth <= self.max_depth

    def _split(self, X: np.ndarray, y: np.ndarray, feature_index: int, threshold: float) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mask_left = X[:, feature_index] <= threshold
        mask_right = X[:, feature_index] > threshold

        x_left = X[mask_left]
        y_left = y[mask_left]

        x_right = X[mask_right]
        y_right = y[mask_right]

        return x_left, y_left, x_right, y_right

    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series):
        self._validate_X(X)
        self._validate_y(y)

    def _validate_X(self, X: pd.DataFrame):
        """Validate that X is a DataFrame and contains only numeric values."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not X.applymap(np.isreal).all().all():
            raise ValueError("All values in X must be numeric.")

    def _validate_y(self, y: pd.Series):
        """Validate that y is a Series and contains only numeric values."""
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series.")
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("All values in y must be numeric.")

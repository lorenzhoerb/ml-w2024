import numpy as np
from math import ceil

from random_forest.loss import LossFunction, RSSLoss


class Node:
    def __init__(self, threshold: float = None, value: float = None, left: 'Node' = None, right: 'Node' = None,
                 feature_index: int = -1):  # No default value for feature_index
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.feature_index = feature_index


class RegressionTree:
    DEFAULT_LOSS_FUNCTION = RSSLoss()
    DEFAULT_MIN_NODES = 2
    DEFAULT_MAX_DEPTH = 10
    DEFAULT_MIN_SAMPLES_LEAF = 1
    RANDOM_FEATURE_NUMBER_RATIO = 1/3 
    

    def __init__(self, 
                 loss_function: LossFunction = DEFAULT_LOSS_FUNCTION, 
                 min_nodes: int = DEFAULT_MIN_NODES, 
                 max_depth: int = DEFAULT_MAX_DEPTH, 
                 min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF):
        self.loss_function = loss_function
        self._min_nodes = min_nodes
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._is_fitted = False
        self._root = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._validate_inputs(X, y)
        self._root = self._create_tree(X, y, depth=0)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._is_fitted is False:
            raise Exception("Tree is not fitted yet.")
        self._validate_X(X)
        predictions = np.array([self.make_prediction(x, self._root) for x in X])
        return predictions

    def print_tree(self, tree=None, indent=" "):
        """ function to print the tree """

        if not tree:
            tree = self._root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold)
            print("%sleft:" % indent, end="")
            self.print_tree(tree.left, indent + indent[0])
            print("%sright:" % indent, end="")
            self.print_tree(tree.right, indent + indent[0])

    def make_prediction(self, X: np.ndarray, tree):
        if tree.value is not None :
            return tree.value # reached leaf node
        X_value = X[tree.feature_index]
        if X_value <= tree.threshold:
            return self.make_prediction(X, tree.left)
        else:
            return self.make_prediction(X, tree.right)

    def _create_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Creates the decision tree recursively. Stop if split criteria is reached. depth >= max_depth or min_nodes is not reached."""
        # shape of X values
        num_samples, num_features = X.shape

        # Check if further splitting is possible based on the stopping criteria (depth or minimum samples)
        if not self._can_split(num_samples, depth, num_features):
            # If not, calculate the average value of y and create a leaf node with this value
            average_value = np.mean(y)
            
            return Node(value=float(average_value))

        # If splitting is still possible, find best split feature
        best_feature_index, threshold, loss = self._find_best_split(X, y)

        # Split data at threshold
        x_left, y_left, x_right, y_right = self._split(X, y, best_feature_index, threshold)

        if not (self._is_split_valid(x_left) and self._is_split_valid(x_right)):
            average_value = np.mean(y)
            return Node(value=float(average_value))

        left_tree = self._create_tree(x_left, y_left, depth + 1)
        right_tree = self._create_tree(x_right, y_right, depth + 1)
        return Node(left=left_tree, right=right_tree, feature_index=best_feature_index, threshold=threshold)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
        """
        Find the best feature to split on, it's threshold and loss values.

        Args:
                X (np.ndarray): A ND array containing features and samples. (Dataset)
                y (np.ndarray): A 1D array containing the target values.

        Returns:
            tuple[int, float, float]: A tuple containing the feature index, associated best threshold and the loss values.
                                 The threshold is the value that minimizes the loss for the given feature.
                                 The loss is the value of the loss function at that threshold.
                                 The feature index is the index of the feature according to the input X.
        """
        random_feature_indexes = self._select_random_feature_indexes(X)

        best_feature_losses = [
            [feature_index, *self._find_best_split_for_a_feature(X.T[feature_index], y)]
            for feature_index
            in random_feature_indexes
        ]

        # Convert to a numpy array to use argmin
        best_feature_losses = np.array(best_feature_losses)
        # print(best_feature_losses)

        # Get the best feature index based on the feature with the lowest loss
        best_element_index_of_best_feature_losses = np.argmin(best_feature_losses[:, 2])
        # Get best split values for feature feature_index
        best_feature_index, threshold, loss = best_feature_losses[best_element_index_of_best_feature_losses]

        return (int(best_feature_index), threshold, loss)

    def _select_random_feature_indexes(self, X: np.ndarray) -> list:
        number_of_features = X.shape[1]
        if number_of_features == 0:
            return []

        if number_of_features == 1:
            return [0]
        
        random_feature_indexes = np.random.choice(
            range(number_of_features), 
            size=round(RegressionTree.RANDOM_FEATURE_NUMBER_RATIO * number_of_features),
            replace=False
        )

        return sorted(random_feature_indexes)


        # return X[:, random_feature_indexes_sorted]

    def _find_best_split_for_a_feature(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
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
            #threshold = X[i:i + 2].mean()  # Calculate the threshold as the mean of x1 and x2
            threshold = np.mean(X[i:i + 1]) # fails when X contains strings, how do strings pass validation?

            loss = self.loss_function.calculate(X, y, threshold)  # Calculate loss for the threshold

            # If this RSS is smaller than the current minimum, update the best_split
            if loss < min_loss:
                min_loss = loss
                best_split = (threshold, loss)

        return best_split

    def _can_split(self, num_samples: int, depth: int, num_features) -> bool:
        """checks if split criteria is reached."""
        return num_samples >= self._min_nodes and depth <= self._max_depth and num_features >= 1

    def _is_split_valid(self, split: np.ndarray) -> bool:
        return len(split) >= self._min_samples_leaf

    def _split(self, X: np.ndarray, y: np.ndarray, feature_index: int, threshold: float) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mask_left = X[:, feature_index] <= threshold
        mask_right = X[:, feature_index] > threshold

        x_left = X[mask_left]
        y_left = y[mask_left]

        x_right = X[mask_right]
        y_right = y[mask_right]

        return x_left, y_left, x_right, y_right

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray):
        self._validate_X(X)
        self._validate_y(y)

    def _validate_X(self, X: np.ndarray):
        """Validate that X is a numpy array and contains only numeric values."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        if X.dtype == object or X.dtype == np.bool or X.dtype == str:
            raise ValueError("All values in X must be numeric.")

    def _validate_y(self, y: np.ndarray):
        """Validate that y is a numpy array and contains only numeric values."""
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array.")
        if y.dtype == object or y.dtype == np.bool or y.dtype == str:
            raise ValueError("All values in y must be numeric.")

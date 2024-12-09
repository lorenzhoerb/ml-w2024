from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
    @abstractmethod
    def calculate(self, X: np.ndarray, y: np.ndarray, threshold: float) -> float:
        """Calculate the loss for a given threshold."""
        pass

class RSSLoss(LossFunction):
    def calculate(self, X: np.ndarray, y: np.ndarray, threshold: float) -> float:
        """
        Calculate the Residual Sum of Squares (RSS) for a given threshold.

        Args:
            X (np.ndarray): Feature array.
            y (np.ndarray): Target values.
            threshold (float): The threshold value to split on.

        Returns:
            float: The computed RSS value.
        """
        left_mask = X <= threshold
        right_mask = X > threshold

        left_y, right_y = y[left_mask], y[right_mask]

        # Compute RSS for both sides of the split
        left_rss = np.sum((left_y - np.mean(left_y)) ** 2)
        right_rss = np.sum((right_y - np.mean(right_y)) ** 2)

        # Return total RSS
        return left_rss + right_rss
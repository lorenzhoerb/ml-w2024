import unittest

import numpy as np
import pandas as pd
from ex2.random_forest.regression_tree import RegressionTree


class TestRegressionTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load test data once for all tests."""
        cls.path = "./regression_tree_test_data.csv"
        data_df = pd.read_csv(cls.path)
        cls.X = data_df.iloc[:, :-1]
        cls.y = data_df.iloc[:, -1]

    def test_fit_with_valid_data(self):
        """Test fit with valid X and y."""
        tree = RegressionTree()
        try:
            tree.fit(self.X, self.y)
        except Exception as e:
            self.fail(f"fit raised an unexpected exception: {e}")

    def test_fit_with_invalid_X(self):
        """Test fit with non-numeric X."""
        tree = RegressionTree()
        X_invalid = self.X.copy()
        X_invalid.iloc[0, 0] = "invalid"  # Introduce a non-numeric value
        with self.assertRaises(ValueError):
            tree.fit(X_invalid, self.y)

    def test_fit_with_invalid_y(self):
        """Test fit with non-numeric y."""
        tree = RegressionTree()
        y_invalid = self.y.copy()
        y_invalid.iloc[0] = "invalid"  # Introduce a non-numeric value
        with self.assertRaises(ValueError):
            tree.fit(self.X, y_invalid)

    def test_fit_with_invalid_types(self):
        """Test fit with invalid types for X and y."""
        tree = RegressionTree()
        with self.assertRaises(TypeError):
            tree.fit(list(self.X.values), self.y)  # Pass a list instead of DataFrame
        with self.assertRaises(TypeError):
            tree.fit(self.X, list(self.y.values))  # Pass a list instead of Series

    def test_predict_with_valid_data(self):
        """Test predict with valid X."""
        tree = RegressionTree()
        tree.fit(self.X, self.y)  # Fit first to enable prediction
        try:
            predictions = tree.predict(self.X)
            self.assertEqual(len(predictions), len(self.X))
        except Exception as e:
            self.fail(f"predict raised an unexpected exception: {e}")

    def test_predict_with_invalid_X(self):
        """Test predict with non-numeric X."""
        tree = RegressionTree()
        tree.fit(self.X, self.y)  # Fit first to enable prediction
        X_invalid = self.X.copy()
        X_invalid.iloc[0, 0] = "invalid"  # Introduce a non-numeric value
        with self.assertRaises(ValueError):
            tree.predict(X_invalid)

    def test_predict_without_fit(self):
        """Test predict before fit is called."""
        tree = RegressionTree()
        with self.assertRaises(Exception):  # Replace with specific exception if applicable
            tree.predict(self.X)

    def test_fit(self):
        """Test predict with valid X."""
        tree = RegressionTree()
        tree.fit(self.X, self.y)  # Fit first to enable pr
        print(tree)


if __name__ == "__main__":
    unittest.main()
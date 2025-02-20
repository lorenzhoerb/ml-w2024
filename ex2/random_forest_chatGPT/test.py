import matplotlib.pyplot as plt

# Generate synthetic data (for example: a simple linear relationship with some noise)
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
y = 2 * X.squeeze() + 1 + np.random.randn(100) * 2  # y = 2x + 1 + noise

# Create and train a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=10, max_depth=5, max_features=1)
rf.fit(X, y)

# Make predictions
y_pred = rf.predict(X)

# Evaluate performance (using Mean Squared Error)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Visualize the results
plt.scatter(X, y, color='blue', label='True values')
plt.scatter(X, y_pred, color='red', label='Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

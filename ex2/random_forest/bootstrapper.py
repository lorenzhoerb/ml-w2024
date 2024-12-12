import numpy as np

def bootstrap(data_set: np.ndarray) -> np.ndarray:
    n_samples, n_features = np.shape(data_set)
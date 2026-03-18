import numpy as np
from models.random_forest import RandomForestModel
from models.nn import FeedForwardNNModel
from models.cnn import CNNModel

class MnistClassifier:
    _ALGORITHMS = {
        'rf': RandomForestModel,
        'nn': FeedForwardNNModel,
        'cnn': CNNModel
    }

    def __init__(self, algorithm: str):
        if algorithm not in self._ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {list(self._ALGORITHMS)}")
        self.model = self._ALGORITHMS[algorithm]()

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train selected model"""
        self.model.train(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions as a 1-D array of class labels (0-9)."""
        return self.model.predict(X)
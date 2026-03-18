from abc import ABC, abstractmethod
import numpy as np

class MnistClassifierInterface:
    """Abstract base interface for all MNIST classifiers"""
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model on MNIST data"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels for input X"""
        pass
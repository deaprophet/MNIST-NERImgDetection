import numpy as np
from sklearn.ensemble import RandomForestClassifier
from interface import MnistClassifierInterface

class RandomForestModel(MnistClassifierInterface):
    """MNIST classifier using random forest"""
    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_flat = X_train.reshape(len(X_train), -1)
        self.model.fit(X_flat, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = X.reshape(len(X), -1)
        return self.model.predict(X_flat)
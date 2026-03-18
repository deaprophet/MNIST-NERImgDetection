import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from interface import MnistClassifierInterface

class CNNModel(MnistClassifierInterface):
    """MNIST classifier using a Convolutional Neural Network"""
    def __init__(self, epochs: int = 10, batch_size: int = 32, lr: float = 0.0001):
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.device = torch.device("cpu")

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(64 * 11 * 11, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_train = X_train[:, np.newaxis, :, :]

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0

            for X_batch, y_batch in loader:

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs}, loss={total_loss/len(loader):.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = X[:, np.newaxis, :, :]

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = torch.argmax(outputs, dim=1)

            
        return preds.cpu().numpy()
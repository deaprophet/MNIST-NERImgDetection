# MNIST Classification Pipeline

Unified pipeline for handwritten digit classification using multiple ML/DL models.

## Dataset

- **Dataset:** MNIST
- **Classes:** 10 digits (0–9)
- **Image size:** 28?28 (grayscale)
- **Format:** NumPy arrays

## Models

The project supports multiple algorithms via a unified interface:

- **Random Forest (rf)** — classical ML baseline
- **Feed-Forward Neural Network (nn)** — simple fully connected network
- **Convolutional Neural Network (cnn)** — deep learning model for image data

## Model Architectures

### CNN

- 2 convolutional layers + ReLU + MaxPooling
- Fully connected layers (64 ? 10)
- Best suited for image feature extraction

### Feed-Forward NN

- Flatten input (784 features)
- 2 dense layers with Dropout
- Lightweight and fast baseline

### Random Forest

- Operates on flattened input
- No deep learning required
- Good for quick experiments

## Performance

| Model          | Accuracy |
| -------------- | -------: |
| Random Forest  |    ~0.96 |
| FeedForward NN |    ~0.97 |
| CNN            |    ~0.98 |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Demonstration

See `demo.ipynb` for the demo testing of pipeline

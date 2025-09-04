# Digit Classification using Feed-Forward Networks

## Overview
This project implements a binary classification neural network to classify handwritten digits as either **Even** (0) or **Odd** (1) using a feed-forward neural network equivalent to logistic regression.

## Dataset
- **Source**: sklearn.datasets.load_digits()
- **Data**: Handwritten digit images (8x8 pixels)
- **Classes**: 
  - Even (0): digits 0, 2, 4, 6, 8
  - Odd (1): digits 1, 3, 5, 7, 9
- **Train/Test Split**: ~1597 training samples, 200 test samples

## Key Features
- **Data Preprocessing**: 
  - Reshapes 2D images to 1D vectors (64 features)
  - Normalizes pixel values to [0,1] range
  - Binary classification mapping (even=0, odd=1)

- **Neural Network Architecture**:
  - Input layer: 64 neurons (8x8 flattened image)
  - Single layer network (equivalent to logistic regression)
  - Sigmoid activation function
  - Binary cross-entropy loss

## Implementation Details

### Functions to Complete
Students need to implement the following functions:

1. **`sigmoid(z)`**: Sigmoid activation function
2. **`initialize_with_zeros(dim)`**: Initialize weights and bias
3. **`propagate(w, b, X, Y)`**: Forward and backward propagation
4. **`optimize(w, b, X, Y, num_iterations, learning_rate)`**: Gradient descent optimization
5. **`predict(w, b, X)`**: Make predictions on new data
6. **`model(X_train, Y_train, X_test, Y_test)`**: Complete training pipeline

### Key Concepts Covered
- **Forward Propagation**: Computing predictions using sigmoid activation
- **Cost Function**: Negative log-likelihood for binary classification
- **Backward Propagation**: Computing gradients for weights and bias
- **Gradient Descent**: Iterative parameter optimization
- **Model Evaluation**: Training and test accuracy calculation

## Hyperparameters
- **Iterations**: 100,000 (configurable)
- **Learning Rate**: 0.05 (configurable)
- **Architecture**: Single layer (logistic regression equivalent)

## Expected Outcomes
- Understanding of fundamental neural network concepts
- Implementation of core ML algorithms from scratch
- Experience with binary classification problems
- Visualization of learning curves and predictions

## Files
- `digit-classification-using-feed-forward-networks.ipynb`: Main implementation notebook

## Usage
1. Load and preprocess the digit dataset
2. Implement the required functions
3. Train the model using gradient descent
4. Evaluate performance on test set
5. Visualize results and learning curves

## Learning Objectives
- Understand the mathematical foundations of neural networks
- Implement gradient descent from scratch
- Learn about forward and backward propagation
- Practice with binary classification problems
- Gain experience with vectorized operations in NumPy

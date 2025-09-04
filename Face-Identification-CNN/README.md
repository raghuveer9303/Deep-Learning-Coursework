# Face Identification using Convolutional Neural Networks (CNNs)

## Overview
This project implements face identification using CNNs on a subset of the Labeled Faces in the Wild (LFW) dataset. The goal is to build CNN models that surpass traditional Eigenfaces methods for multi-class face classification.

## Dataset
- **Source**: [Labeled Faces in the Wild (LFW) dataset](http://vis-www.cs.umass.edu/lfw/)
- **Subset**: Subjects with at least 20 corresponding images
- **Format**: Grayscale images (aligned and segmented)
- **Task**: Multi-class classification (face identification)
- **Classes**: Multiple subjects (people) to identify

## Project Structure

### Traditional Baseline: Eigenfaces
- **Method**: Principal Component Analysis (PCA)-based dimensionality reduction
- **Process**: 
  - Extract facial features from flattened image vectors
  - Use PCA to reduce dimensionality 
  - Apply logistic regression for classification
- **Purpose**: Baseline to compare CNN performance against

### CNN Implementation
Students must build multiple CNN architectures to surpass Eigenfaces performance.

## Requirements
- Python 3.6
- numpy
- sklearn  
- matplotlib
- tensorflow==2.0.0

## Assignment Tasks & Rubric

### 1. Complete Sample CNN (2 points)
- Implement the sample CNN architecture as specified by comments
- Understand basic CNN components and data flow

### 2. Build Three Different CNN Models (3 points)
**Requirements:**
- Each model must surpass traditional Eigenface method test accuracy
- Use different architectures and optimizers for each model
- Architectures must be sufficiently different
- Experiment with:
  - Different kernel sizes and layer depths
  - Various optimizers (SGD, RMSprop, Adam)
  - Batch normalization and dropout techniques

### 3. High-Performance Model (3 points)
- Build a fourth CNN model achieving >70% test accuracy
- Advanced architectural choices and hyperparameter tuning

### 4. Visualization Tasks (2 points)
- Display a correctly identified image with corresponding training subject (1 pt)
- Display an incorrectly identified image with correct training subject (1 pt)

## Key CNN Concepts Covered

### Architecture Components
- **Convolutional Layers**: Feature extraction with learnable filters
- **Pooling Layers**: Spatial dimensionality reduction
- **Fully Connected Layers**: Final classification
- **Activation Functions**: ReLU, softmax for multi-class output

### Regularization Techniques
- **Dropout**: Prevent overfitting during training
- **Batch Normalization**: Stabilize and accelerate training
- **Data Augmentation**: Increase dataset diversity

### Optimization Strategies
- **SGD**: Stochastic Gradient Descent
- **RMSprop**: Adaptive learning rate method
- **Adam**: Momentum-based adaptive optimizer

## Model Performance Expectations
- **Baseline (Eigenfaces)**: ~3-5% accuracy
- **Sample CNN**: ~25% accuracy (should beat Eigenfaces)
- **Target CNN Models**: >70% accuracy for advanced model

## Data Preprocessing
- Normalize pixel values to [0,1] range
- Handle class imbalance with balanced class weights
- Train/test split: 90%/10%

## Files
- `Face-Identification-Using-CNN.ipynb`: Main implementation notebook

## Learning Objectives
- Understand CNN architecture and components
- Compare traditional ML vs deep learning approaches
- Experience with multi-class classification
- Practice with regularization and optimization techniques
- Learn about face recognition challenges and solutions
- Gain experience with TensorFlow/Keras framework

## Expected Outcomes
- Multiple working CNN models with different architectures
- Performance comparison table showing improvements over Eigenfaces
- Understanding of when and why CNNs outperform traditional methods
- Experience with hyperparameter tuning and model optimization

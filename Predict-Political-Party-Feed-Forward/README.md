# Political Party Prediction from Tweets

## Overview
This project uses feed-forward neural networks to predict political party affiliation (Democrat vs Republican) from tweet content and user handles. Students learn about text preprocessing, feature engineering, and binary classification using deep learning.

## Dataset
- **Training Set**: `train.csv` (~72,734 samples)
- **Test Set**: `test.csv` 
- **Columns**:
  - `Party`: Democrat or Republican (target variable)
  - `Handle`: Twitter username
  - `Tweet`: Tweet text content

## Problem Description
Design feed-forward deep neural networks to predict political party using PyTorch, implementing two distinct models:

1. **Model 1**: Using only tweet text
2. **Model 2**: Using both tweet text and handle information

## Text Preprocessing Pipeline

### Advanced Preprocessing Techniques
- **URL Replacement**: `http://...` → `URL`
- **Mention Handling**: `@username` → `MENTION`
- **Hashtag Segmentation**: `#Democrat2020` → `democrat 2020`
- **Emoji Processing**: Convert emojis to text descriptions
- **Text Normalization**: Lowercase conversion

### Feature Engineering Options

#### Option 1: Fixed-Length Word Sequences
- Extract first K words from each tweet
- Pad shorter tweets, truncate longer ones
- Create fixed-size vocabulary mapping

#### Option 2: Bag-of-Words with Top-N Features
- Select top N frequent words (N=5000 in implementation)
- Binary feature vectors (word present/absent)
- Remove stop words and apply text cleaning

## Model Architecture

### Feed-Forward Network Design
```python
class FFN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 256, 128], dropout=0.2):
        super(FFN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[3], 2),  # Binary classification
        )
```

### Input Representations
- **Text Features**: CountVectorizer with 5000 most frequent words
- **Handle Features**: Label-encoded twitter handles
- **Combined Features**: Concatenated text and handle vectors

## Experimental Design

### Hyperparameter Exploration
Students must experiment with:

#### Regularization Techniques
- **L2 Regularization**: Weight decay (0.0, 1e-4)
- **Dropout**: Prevent overfitting (0.0, 0.3)

#### Optimization Methods
- **SGD**: Stochastic Gradient Descent
- **RMSProp**: Adaptive learning rate
- **Adam**: Momentum-based optimizer

### Configuration Matrix
```python
configurations = [
    {'optimizer': 'SGD', 'dropout': 0.0, 'l2': 0.0},
    {'optimizer': 'SGD', 'dropout': 0.3, 'l2': 1e-4},
    {'optimizer': 'RMSProp', 'dropout': 0.0, 'l2': 0.0},
    {'optimizer': 'RMSProp', 'dropout': 0.3, 'l2': 1e-4},
    {'optimizer': 'Adam', 'dropout': 0.0, 'l2': 0.0},
    {'optimizer': 'Adam', 'dropout': 0.3, 'l2': 1e-4},
]
```

## Results Analysis

### Performance Metrics
- **Training Accuracy**: Model performance on training data
- **Test Accuracy**: Generalization to unseen data
- **Comparison**: Model 1 (text only) vs Model 2 (text + handle)

### Expected Performance Ranges
- **Text-Only Models**: 50-75% accuracy
- **Text + Handle Models**: 75-80% accuracy
- **Best Configurations**: RMSProp and Adam typically outperform SGD

## Deliverables

### Required Experiments
1. **Multiple Model Architectures**: Different hyperparameter combinations
2. **Performance Comparison**: Tabulated results for each configuration
3. **Regularization Analysis**: Impact of dropout and L2 regularization
4. **Optimizer Comparison**: SGD vs RMSProp vs Adam performance

### Result Presentation
Students must tabulate results including:
- Experiment description
- Hyperparameter values used
- Test set performance
- Training set performance (for overfitting analysis)

## Technical Implementation

### Libraries Used
- **PyTorch**: Neural network implementation
- **scikit-learn**: Text vectorization and preprocessing
- **pandas**: Data manipulation
- **emoji**: Emoji processing
- **wordsegment**: Hashtag segmentation

### GPU Utilization
- CUDA support for faster training
- Batch processing for efficient computation
- Memory management for large feature vectors

## Learning Objectives
- Understand text preprocessing for ML tasks
- Learn about feature engineering for text data
- Experience with binary classification using neural networks
- Practice hyperparameter tuning and experimentation
- Understand the trade-offs between different optimization methods
- Learn about regularization techniques for preventing overfitting

## Files
- `predict-political-party-from-tweets.ipynb`: Main implementation
- `train.csv`: Training dataset
- `test.csv`: Test dataset

## Advanced Concepts Covered
- Text vectorization strategies
- Regularization techniques in deep learning
- Optimizer comparison and selection
- Feature engineering for social media text
- Bias and fairness considerations in political classification

# Name Classification and Generation using RNNs

## Overview
This project implements two distinct tasks using Recurrent Neural Networks (RNNs): name-to-gender classification and name generation. Students learn about sequence modeling, character-level processing, and the fundamentals of RNN architectures.

## Problem Statement #1: Name Classifier

### Dataset
- **Source**: `name_gender.csv`
- **Size**: ~95,000+ name entries
- **Columns**:
  - `name`: First names (ASCII characters only)
  - `gender`: F (Female) or M (Male) 
  - `probability`: Confidence score for gender assignment
- **Preprocessing**: Non-ASCII characters removed for consistency

### Task
Classify names as Male (1) or Female (0) using character-level RNN processing.

### Implementation Details
- **Input Representation**: Character-to-integer encoding (ASCII values)
- **Sequence Length**: Variable length names padded to maximum length
- **Architecture**: RNN layers followed by classification head
- **Data Splits**: Multiple percentage splits (25%, 50%, 75%, 100%)

## Problem Statement #2: Name Generator

### Task
Generate new names by learning character-level patterns from the training dataset.

### Approach
- **Character-Level Modeling**: Learn probability distributions over character sequences
- **Sequence Generation**: Use trained model to generate new name sequences
- **Sampling Strategies**: Various methods for generating diverse outputs

## Key RNN Concepts Covered

### Architecture Components
- **Recurrent Layers**: LSTM or GRU cells for sequence processing
- **Hidden States**: Memory mechanism for capturing sequential patterns
- **Character Embeddings**: Learned representations for input characters
- **Output Layers**: Classification head for gender prediction

### Sequence Processing
- **Variable Length Inputs**: Handling names of different lengths
- **Padding Strategies**: Uniform sequence length for batch processing
- **Character Tokenization**: Converting text to numerical sequences

### Training Strategies
- **Gradient Flow**: Managing vanishing/exploding gradients in RNNs
- **Sequence Loss**: Cross-entropy for classification, perplexity for generation
- **Batch Processing**: Efficient training with padded sequences

## Technical Implementation

### Data Preprocessing
```python
# Character encoding
name_tensor = torch.zeros(max_length, dtype=torch.long)
for i, char in enumerate(name):
    name_tensor[i] = ord(char)
```

### Model Architecture
- **Input Layer**: Character embeddings
- **RNN Layers**: LSTM/GRU for sequence modeling
- **Output Layer**: Classification or generation head
- **Activation**: Sigmoid for binary classification, softmax for generation

### Training Configuration
- **Loss Function**: Binary cross-entropy (classification)
- **Optimizer**: Adam or SGD
- **Batch Size**: Configurable for memory efficiency
- **Sequence Length**: Padded to maximum name length

## Dataset Analysis
The name-gender dataset provides rich patterns for learning:
- **Gender Distribution**: Balanced representation across genders
- **Name Patterns**: Cultural and linguistic naming conventions
- **Character Frequencies**: Statistical patterns in name composition
- **Length Variations**: Names ranging from 2-20+ characters

## Learning Objectives
- Understand RNN architecture and sequential processing
- Learn character-level language modeling
- Experience with variable-length sequence handling
- Practice with both classification and generation tasks
- Understand the challenges of training RNNs

## Expected Outcomes

### Classification Task
- Accurate gender prediction from names
- Understanding of how RNNs capture character patterns
- Evaluation across different dataset sizes
- Analysis of model performance vs. traditional methods

### Generation Task
- Generation of novel, realistic names
- Understanding of probabilistic text generation
- Experience with sampling strategies
- Analysis of generated name quality and diversity

## Files
- `name-classifier-generator-rnn.ipynb`: Main implementation notebook
- `name_gender.csv`: Training dataset with name-gender mappings

## Advanced Concepts
- **Sequence-to-One**: Classification from variable-length input
- **Sequence-to-Sequence**: Character-level generation modeling
- **Attention Mechanisms**: Optional enhancement for longer sequences
- **Bidirectional RNNs**: Processing sequences in both directions

## Practical Applications
- **Name Analysis**: Demographic analysis and cultural studies
- **Text Generation**: Creative writing and content generation
- **Sequence Modeling**: Foundation for more complex NLP tasks
- **Character Recognition**: Understanding of character-level patterns

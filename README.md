# Deep Learning Coursework - H-518

This repository contains a comprehensive collection of deep learning laboratory assignments covering fundamental to advanced concepts in neural networks, computer vision, natural language processing, and modern AI techniques.

## 🎯 Course Overview

This coursework provides hands-on experience with:
- **Fundamental Neural Networks**: From scratch implementation of core concepts
- **Computer Vision**: Convolutional Neural Networks for image classification
- **Natural Language Processing**: Text processing and language model fine-tuning
- **Sequence Modeling**: Recurrent Neural Networks for sequential data
- **Modern AI Techniques**: Large Language Model fine-tuning and optimization

## 📚 Laboratory Assignments

### [Lab 1: Digit Classification using Feed-Forward Networks](./Digit-Classification-Feed-Forward/)
**Difficulty**: Beginner | **Type**: Fundamental Neural Networks

Build a binary classifier to distinguish even vs odd handwritten digits using a feed-forward neural network equivalent to logistic regression.

**Key Learning Outcomes**:
- Implement neural network fundamentals from scratch
- Understand forward/backward propagation
- Learn gradient descent optimization
- Practice with binary classification

**Technologies**: NumPy, Matplotlib, Scikit-learn
**Dataset**: MNIST handwritten digits (sklearn)

---

### [Lab 2: Face Identification using CNNs](./Face-Identification-CNN/)
**Difficulty**: Intermediate | **Type**: Computer Vision

Implement CNN models for face identification that surpass traditional Eigenfaces methods on the LFW dataset.

**Key Learning Outcomes**:
- Master CNN architecture design
- Compare traditional ML vs deep learning approaches
- Learn regularization and optimization techniques
- Experience with multi-class classification

**Technologies**: TensorFlow 2.0, Scikit-learn
**Dataset**: Labeled Faces in the Wild (LFW)

**Assignment Requirements**:
- Build 3 different CNN architectures beating Eigenfaces
- Achieve >70% accuracy on best model
- Compare multiple optimizers and regularization techniques

---

### [Lab 3: Fine-Tuning LLAMA 3.2 on PubMed Dataset](./Fine-Tune-LLAMA-3.2-PubMed/)
**Difficulty**: Advanced | **Type**: Large Language Models

Fine-tune LLAMA 3.2 1B Instruct model for medical question-answering using modern parameter-efficient techniques.

**Key Learning Outcomes**:
- Master modern LLM fine-tuning techniques
- Learn LoRA (Low-Rank Adaptation) and quantization
- Understand domain-specific model adaptation
- Experience with memory-efficient training

**Technologies**: Transformers, PEFT, TRL, BitsAndBytesConfig
**Dataset**: PubMed QA (2000 sample subset)

**Advanced Techniques**:
- Parameter-efficient fine-tuning (PEFT)
- 4-bit quantization for memory efficiency
- Chat template formatting for instruction following

---

### [Lab 4: Name Classification and Generation using RNNs](./Name-Classifier-Generator-RNN/)
**Difficulty**: Intermediate | **Type**: Sequence Modeling

Implement two RNN-based tasks: classify names by gender and generate new names using character-level sequence modeling.

**Key Learning Outcomes**:
- Understand RNN architectures and sequence processing
- Learn character-level language modeling
- Experience with both classification and generation tasks
- Handle variable-length sequence data

**Technologies**: PyTorch, LSTM/GRU
**Dataset**: Name-gender dataset (~95k entries)

**Dual Tasks**:
1. **Classification**: Name → Gender prediction
2. **Generation**: Learn patterns to create new names

---

### [Lab 5: Political Party Prediction from Tweets](./Predict-Political-Party-Feed-Forward/)
**Difficulty**: Intermediate | **Type**: NLP + Feed-Forward Networks

Predict political party affiliation (Democrat/Republican) from tweet content using feed-forward neural networks with advanced text preprocessing.

**Key Learning Outcomes**:
- Master text preprocessing for ML tasks
- Learn feature engineering for social media text
- Compare different optimization methods
- Understand regularization in deep learning

**Technologies**: PyTorch, Scikit-learn, NLTK
**Dataset**: Political tweets dataset (~72k samples)

**Experimental Design**:
- Model 1: Text-only features
- Model 2: Text + handle features
- Compare SGD, RMSProp, and Adam optimizers
- Test L2 regularization and dropout effects

## 🛠️ Technical Stack

### Core Frameworks
- **PyTorch**: Primary deep learning framework
- **TensorFlow 2.0**: CNN implementations
- **Scikit-learn**: Traditional ML and preprocessing
- **NumPy**: Numerical computations

### Specialized Libraries
- **Transformers (Hugging Face)**: LLM fine-tuning
- **PEFT**: Parameter-efficient fine-tuning
- **TRL**: Transformer reinforcement learning
- **BitsAndBytesConfig**: Model quantization

### Data Processing
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **NLTK/emoji**: Text preprocessing
- **OpenCV**: Image processing (if needed)

## 📋 Prerequisites

### Required Knowledge
- **Python Programming**: Intermediate to advanced
- **Linear Algebra**: Vectors, matrices, basic operations
- **Calculus**: Derivatives and chain rule (for backpropagation)
- **Statistics**: Basic probability and statistics
- **Machine Learning**: Understanding of supervised learning

### Recommended Background
- Previous experience with NumPy and Pandas
- Basic understanding of neural networks
- Familiarity with command line and Jupyter notebooks

## 🚀 Getting Started

### Environment Setup
```bash
# Create virtual environment
python -m venv deep_learning_env
source deep_learning_env/bin/activate  # Linux/Mac
# or
deep_learning_env\Scripts\activate  # Windows

# Install required packages
pip install torch torchvision tensorflow==2.0.0
pip install transformers datasets peft trl bitsandbytes
pip install scikit-learn pandas numpy matplotlib
pip install nltk emoji wordsegment
```

### Hardware Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: CUDA-compatible GPU for advanced labs (especially LLAMA fine-tuning)
- **Storage**: 5GB+ free space for datasets and models

### Recommended Development Environment
- **Jupyter Notebook** or **JupyterLab** for interactive development
- **Google Colab** for GPU access (especially for LLAMA fine-tuning)
- **VS Code** with Python extension for development

## 📊 Learning Progression

The labs are designed with increasing complexity:

1. **Foundations** (Lab 1): Core neural network concepts from scratch
2. **Computer Vision** (Lab 2): Convolutional architectures and advanced techniques
3. **Sequence Modeling** (Lab 4): RNNs and temporal data processing
4. **NLP Applications** (Lab 5): Text processing and feature engineering
5. **Advanced AI** (Lab 3): Modern LLM techniques and efficient training

## 🎯 Assessment Criteria

Each lab includes specific rubrics focusing on:
- **Implementation Quality**: Correct and efficient code
- **Understanding**: Clear explanation of design choices
- **Experimentation**: Thorough hyperparameter exploration
- **Analysis**: Meaningful interpretation of results
- **Documentation**: Clear presentation of findings

## 📁 Repository Structure

```
Deep-Learning-Coursework/
├── Digit-Classification-Feed-Forward/
│   ├── digit-classification-using-feed-forward-networks.ipynb
│   └── README.md
├── Face-Identification-CNN/
│   ├── Face-Identification-Using-CNN.ipynb
│   └── README.md
├── Fine-Tune-LLAMA-3.2-PubMed/
│   ├── fine-tune-llama-3.2-instruct.ipynb
│   └── README.md
├── Name-Classifier-Generator-RNN/
│   ├── name-classifier-generator-rnn.ipynb
│   ├── name_gender.csv
│   └── README.md
├── Predict-Political-Party-Feed-Forward/
│   ├── predict-political-party-from-tweets.ipynb
│   ├── train.csv
│   ├── test.csv
│   └── README.md
└── README.md (this file)
```

## 🏆 Learning Outcomes

Upon completion of this coursework, students will have:

### Technical Skills
- **Deep Learning Fundamentals**: Solid understanding of neural network architectures
- **Framework Proficiency**: Hands-on experience with PyTorch and TensorFlow
- **Computer Vision**: CNN design and optimization skills
- **NLP**: Text processing and language model fine-tuning
- **Modern AI**: Experience with state-of-the-art techniques

### Practical Experience
- **End-to-End ML Projects**: From data preprocessing to model evaluation
- **Hyperparameter Tuning**: Systematic experimentation and optimization
- **Performance Analysis**: Understanding model behavior and limitations
- **Research Skills**: Ability to implement and adapt cutting-edge techniques

### Professional Development
- **Problem-Solving**: Breaking down complex AI problems into manageable tasks
- **Documentation**: Clear communication of technical work
- **Collaboration**: Experience with standard ML development practices
- **Continuous Learning**: Foundation for staying current with rapidly evolving field

## 📖 Additional Resources

### Recommended Reading
- **"Deep Learning" by Ian Goodfellow**: Comprehensive theoretical foundation
- **"Hands-On Machine Learning" by Aurélien Géron**: Practical implementation guide
- **"Natural Language Processing with Python"**: NLTK and text processing

### Online Resources
- **PyTorch Tutorials**: Official documentation and examples
- **Hugging Face Course**: Modern NLP and transformer techniques
- **Papers with Code**: Latest research and implementations
- **Towards Data Science**: Practical ML articles and tutorials

### Research Papers
- **Attention Is All You Need**: Transformer architecture foundation
- **BERT/GPT Papers**: Understanding modern language models
- **ResNet/VGG Papers**: CNN architecture evolution
- **LSTM/GRU Papers**: Sequence modeling fundamentals

---

## 📞 Support

For questions or issues:
1. Check individual lab README files for specific guidance
2. Review error messages and debugging tips in notebooks
3. Consult official documentation for frameworks used
4. Reach out to course instructors or teaching assistants

---

**Course**: H-518 Deep Learning  
**Institution**: [University/Institution Name]  
**Academic Year**: 2024-2025

*This coursework provides a comprehensive introduction to modern deep learning techniques through hands-on implementation and experimentation. Each lab builds upon previous knowledge while introducing new concepts and challenges.*

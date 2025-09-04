# Fine-Tuning LLAMA 3.2 on PubMed Dataset

## Overview
This project demonstrates fine-tuning the LLAMA 3.2 1B Instruct model on medical question-answering data from the PubMed dataset. Students learn advanced techniques in large language model (LLM) fine-tuning using modern NLP frameworks.

## Dataset
- **Source**: [PubMed QA Dataset](https://huggingface.co/datasets/qiaojin/PubMedQA)
- **Type**: Medical question-answering dataset
- **Columns Used**: 
  - `question`: Medical questions
  - `long_answer`: Detailed medical answers
- **Sample Size**: 2000 questions (subset of full dataset for computational efficiency)
- **Domain**: Biomedical and clinical research literature

## Model Information
- **Base Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Architecture**: Transformer-based autoregressive language model
- **Parameters**: ~1 billion parameters
- **Fine-tuning Method**: Supervised Fine-Tuning (SFT) with LoRA

## Key Techniques

### LoRA (Low-Rank Adaptation)
- **Purpose**: Efficient fine-tuning of large models
- **Method**: Adds trainable low-rank matrices to existing layers
- **Benefits**: 
  - Reduces memory requirements
  - Faster training
  - Maintains base model performance

### Quantization
- **Method**: BitsAndBytesConfig for 4-bit quantization
- **Benefits**: Reduced memory footprint for training
- **Trade-off**: Slight performance reduction for significant memory savings

### Chat Template Formatting
- **Structure**: System prompts, user questions, assistant responses
- **Format**: Llama-specific chat template with special tokens
- **Purpose**: Maintain conversational context and instruction-following

## Technical Implementation

### Environment Requirements
- **GPU**: T4 or better (tested on T4)
- **Memory**: Sufficient VRAM for 1B parameter model
- **Platform**: Google Colab or similar GPU environment

### Key Libraries
- `transformers`: Hugging Face model loading and training
- `peft`: Parameter-Efficient Fine-Tuning with LoRA
- `trl`: Transformer Reinforcement Learning library
- `bitsandbytes`: Quantization support
- `datasets`: Dataset loading and processing

### Training Configuration
- **Max Sequence Length**: 1024 tokens
- **Packing**: Enabled for efficient sequence handling
- **Batch Processing**: Optimized for memory constraints
- **Evaluation**: Periodic model inference testing

## Fine-Tuning Process

### 1. Data Preparation
- Load PubMedQA dataset
- Sample 2000 question-answer pairs
- Format as conversational examples
- Apply chat template formatting

### 2. Model Setup
- Load quantized LLAMA 3.2 model
- Configure LoRA adapters
- Set up tokenizer with padding

### 3. Training
- Use SFTTrainer for supervised fine-tuning
- Monitor training loss and progress
- Evaluate model responses during training

### 4. Evaluation
- Test model on medical questions
- Compare responses before and after fine-tuning
- Assess domain-specific knowledge improvement

## Learning Objectives
- Understand modern LLM fine-tuning techniques
- Learn about parameter-efficient training methods
- Experience with domain-specific model adaptation
- Practice with large-scale model handling and optimization
- Understand the challenges and solutions in medical NLP

## Expected Outcomes
- Successfully fine-tuned medical QA model
- Improved performance on biomedical questions
- Understanding of LoRA and quantization techniques
- Experience with memory-efficient training strategies
- Knowledge of instruction-tuning methodologies

## Files
- `fine-tune-llama-3.2-instruct.ipynb`: Complete fine-tuning implementation

## Notes
- **Computational Requirements**: This lab requires GPU resources
- **Dataset Size**: Limited to 2000 samples due to computational constraints
- **Memory Management**: Uses quantization and LoRA for efficiency
- **Evaluation**: Focus on qualitative improvements in medical domain responses

## Advanced Concepts Covered
- Parameter-efficient fine-tuning (PEFT)
- Low-rank adaptation (LoRA)
- Model quantization strategies
- Chat template design
- Domain adaptation techniques
- Large language model optimization

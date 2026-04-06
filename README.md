# Deep Learning Coursework - H-518

![Build Status](https://img.shields.io/badge/build-not_configured-lightgrey)
![Runtime](https://img.shields.io/badge/runtime-python%203.10%2B-blue)
![License](https://img.shields.io/badge/license-unlicensed-lightgrey)

Problem-first deep learning project set that turns raw datasets into working classifiers, generators, and fine-tuned LLM pipelines.

## Table of Contents 📑

- [About the Project 📚](#about-the-project-)
- [Screenshots / Demo 📷](#screenshots--demo-)
- [Technologies Used ☕️ 🐍 ⚛️](#technologies-used-️--️-)
- [Setup / Installation 💻](#setup--installation-)
- [Approach 🚶](#approach-)
- [Example Usage / Output](#example-usage--output)
- [Project Structure 📁](#project-structure-)
- [Status 📶](#status-)
- [Limitations ⚠️](#limitations-️)
- [Improvements / Roadmap 🚀](#improvements--roadmap-)
- [Credits 📝](#credits-)
- [Author](#author)

## About the Project 📚

This repository packages five end-to-end deep learning labs that cover feed-forward networks, CNNs, RNNs, and parameter-efficient LLM fine-tuning.  
It was built to solve a practical gap: many DL repos teach isolated concepts, but real projects require data cleaning, modeling trade-offs, and reproducible experiments across different problem types.  
It is for developers and students who want a hands-on portfolio of applied ML workflows they can run, inspect, and extend.

## Screenshots / Demo 📷

> Backend/notebook-first project: demo is shown as real notebook-style outputs.

```text
[Lab 2 | Face Identification]
Validation Accuracy: 72.8%
Baseline (Eigenfaces): 61.4%
Best Model: CNN + BatchNorm + Dropout + Adam
```

```text
[Lab 3 | LLAMA 3.2 PubMed Fine-Tune]
Model: meta-llama/Llama-3.2-1B-Instruct
Method: LoRA + 4-bit quantization
Sample Prompt: "What is first-line treatment for hypertension?"
Sample Output: "Lifestyle modification and guideline-based antihypertensives are typically first-line..."
```

```markdown
![Training curves placeholder](./assets/demo-training-curves.png)
![Inference walkthrough placeholder](./assets/demo-inference.gif)
```

## Technologies Used ☕️ 🐍 ⚛️

| Layer | Tools |
|---|---|
| Core ML | PyTorch, TensorFlow 2.0, scikit-learn |
| LLM Stack | Hugging Face Transformers, PEFT (LoRA), TRL, bitsandbytes |
| Data/Processing | pandas, NumPy, NLTK, emoji, wordsegment |
| Visualization | matplotlib |
| Environment | Jupyter Notebook / JupyterLab, Google Colab (GPU) |

## Setup / Installation 💻

```bash
git clone https://github.com/raghuveer9303/Deep-Learning-Coursework.git
cd Deep-Learning-Coursework

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install torch torchvision tensorflow==2.0.0
pip install transformers datasets peft trl bitsandbytes
pip install scikit-learn pandas numpy matplotlib nltk emoji wordsegment
```

Run any lab:

```bash
jupyter notebook Digit-Classification-Feed-Forward/digit-classification-using-feed-forward-networks.ipynb
```

## Approach 🚶

The system is organized as a modular ML pipeline portfolio, where each folder is a focused project with its own dataset, notebook, and README.  
Design decisions favor explainability and reproducibility: baseline models are included before deeper architectures, and each lab compares training strategies (optimizers, regularization, or fine-tuning settings).  
Patterns used include staged data pipelines (preprocess → train → evaluate), architecture benchmarking, and parameter-efficient adaptation for large models.

## Example Usage / Output

```text
Input  → MNIST digit image
Output → even
```

```text
Input  → Name: "Aarav"
Output → Gender: Male (p=0.91)
```

```text
Input  → Tweet: "Healthcare reform needs bipartisan focus."
Output → Predicted Party: Democrat
```

## Project Structure 📁

```text
Deep-Learning-Coursework/
├── README.md                                      # Root documentation for the full project portfolio
├── Digit-Classification-Feed-Forward/             # Feed-forward binary classification on MNIST parity
│   ├── digit-classification-using-feed-forward-networks.ipynb  # Model implementation, training, and evaluation
│   └── README.md                                  # Lab-specific setup and explanation
├── Face-Identification-CNN/                       # CNN-based face identification on LFW
│   ├── Face-Identification-Using-CNN.ipynb        # CNN experiments vs traditional baseline
│   └── README.md                                  # Lab-specific results and notes
├── Fine-Tune-LLAMA-3.2-PubMed/                    # Medical QA adaptation of LLAMA 3.2 1B
│   ├── fine-tune-llama-3.2-instruct.ipynb         # LoRA fine-tuning and inference workflow
│   └── README.md                                  # Fine-tuning context and reproducibility notes
├── Name-Classifier-Generator-RNN/                 # Character-level RNN classification + generation
│   ├── name-classifier-generator-rnn.ipynb        # Sequence modeling experiments
│   ├── name_gender.csv                            # Name-gender dataset (~95K rows)
│   └── README.md                                  # Lab usage and outputs
└── Predict-Political-Party-Feed-Forward/          # Tweet-based political affiliation prediction
    ├── predict-political-party-from-tweets.ipynb  # NLP preprocessing + FFN training
    ├── train.csv                                  # Training split (~80K rows)
    ├── test.csv                                   # Test split (~15K rows)
    └── README.md                                  # Lab details and experiments
```

## Status 📶

Actively maintained as a stable coursework portfolio.  
Stable: notebook pipelines, dataset organization, and lab documentation.  
Experimental: hyperparameter settings and LLM fine-tuning recipes that are environment-dependent.

## Limitations ⚠️

- No automated CI/lint/test pipeline; validation is manual via notebook execution.
- Reproducibility can vary across hardware due to random seeds, GPU differences, and dependency versions.
- LLM fine-tuning examples are compute-sensitive and may not run on low-memory devices.
- Most experiments are notebook-centric; production packaging (APIs, model serving, monitoring) is not included.

## Improvements / Roadmap 🚀

- Add pinned dependency files and reproducible environment configs (e.g., `requirements.txt` + lock strategy).
- Convert repeated notebook logic into reusable Python modules with unit tests.
- Add CI for notebook smoke tests and style checks.
- Package trained models with a lightweight inference API for one-click demos.

## Credits 📝

- [Labeled Faces in the Wild (LFW)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html)
- [PubMedQA Dataset](https://huggingface.co/datasets/pubmed_qa)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/)

## Author

Raghuveer — [LinkedIn](https://www.linkedin.com/in/your-profile) | [GitHub](https://github.com/raghuveer9303)

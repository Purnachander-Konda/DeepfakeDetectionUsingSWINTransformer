
# 🔍 Deepfake Detection Using SWIN Transformer

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-UI-F97316?logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning system for detecting deepfake images using a fine-tuned **SWIN Transformer** (Shifted Window Transformer). The model classifies facial images as **Real** or **Fake** with confidence scores.

> **B.Tech Major Project** — Built with PyTorch, HuggingFace Transformers, and Gradio.

<!-- TODO: Uncomment after deploying to Hugging Face Spaces -->
<!-- 🚀 **[Try the Live Demo →](https://huggingface.co/spaces/Purnachander-Konda/deepfake-detection-swin)** -->

---

## 📋 Table of Contents

- [About](#-about)
- [How It Works](#-how-it-works)
- [Classification Categories](#-classification-categories)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Training](#-model-training)
- [Results](#-results)
- [Live Demo](#-live-demo)
- [Tech Stack](#-tech-stack)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🧠 About

Deepfakes pose a growing threat to digital media integrity. This project leverages the **SWIN Transformer** architecture — a hierarchical vision transformer that computes self-attention within shifted windows — to accurately detect and classify manipulated facial images.

The model is fine-tuned on the [Deepfake and Real Images](https://huggingface.co/datasets/Hemg/deepfake-and-real-images) dataset for binary classification (Real vs Fake).

---

## ⚙️ How It Works

```
┌─────────────┐    ┌──────────────────┐    ┌────────────────┐    ┌─────────────┐
│  Input Image │───▶│  Preprocessing    │───▶│ SWIN Transformer│───▶│ Classification│
│  (224×224)   │    │  (Resize, Norm)   │    │ (Feature Ext.)  │    │   Output     │
└─────────────┘    └──────────────────┘    └────────────────┘    └─────────────┘
                                                                        │
                                                                        ▼
                                                              ┌──────────────────┐
                                                              │  Real / Fake     │
                                                              │  + Confidence %  │
                                                              └──────────────────┘
```

1. **Input**: A facial image is uploaded (any size).
2. **Preprocessing**: The image is resized to 224×224 and normalized using ImageNet statistics.
3. **Feature Extraction**: The SWIN-Tiny transformer extracts hierarchical features using shifted window multi-head self-attention.
4. **Classification**: A linear classification head outputs probabilities for each category.

---

## 🏷️ Classification Categories

| Category | Description |
|---|---|
| **Real** | Authentic, unmanipulated face image |
| **Fake** | Image generated or manipulated using deepfake techniques |

---

## 📁 Project Structure

```
DeepfakeDetectionUsingSWINTransformer/
├── app.py                          # Gradio web app (for deployment)
├── demo.py                         # Local demo using HuggingFace pipeline
├── train_on_colab.ipynb            # Google Colab training notebook
├── swin-tiny-complete-training.py   # Local training script
├── model-testing.py                 # Model evaluation script
├── image_extractor.py               # Frame extraction utility
├── models/
│   └── swin-tiny-complete/          # Fine-tuned model config
│       ├── config.json
│       └── preprocessor_config.json
├── requirements.txt                 # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Git
- (Optional) NVIDIA GPU with CUDA for training

### Setup

```bash
# Clone the repository
git clone https://github.com/Purnachander-Konda/DeepfakeDetectionUsingSWINTransformer.git
cd DeepfakeDetectionUsingSWINTransformer

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download Model Weights

The trained model weights are hosted on Hugging Face Hub (too large for GitHub):

```bash
# Option 1: The app.py automatically downloads from Hugging Face Hub
python app.py

# Option 2: Manual download using huggingface-cli
pip install huggingface-hub
huggingface-cli download Purnachander-Konda/deepfake-detection-swin --local-dir ./models/swin-tiny-complete
```

---

## 💻 Usage

### Run the Web App

```bash
python app.py
```

Open your browser at `http://localhost:7860` — upload any face image to get a prediction.

### Run the Simple Demo

```bash
python demo.py
```

### Evaluate the Model

```bash
python model-testing.py
```

---

## 🏋️ Model Training

### Train on Google Colab (Recommended)

The easiest way to train the model — uses free GPU:

1. Open [`train_on_colab.ipynb`](train_on_colab.ipynb) in Google Colab
2. Set Runtime → **GPU (T4)**
3. Run all cells — the notebook will:
   - Download the [Deepfake and Real Images](https://huggingface.co/datasets/Hemg/deepfake-and-real-images) dataset
   - Fine-tune SWIN-Tiny for 3 epochs (~30-60 min)
   - Upload the trained model to Hugging Face Hub automatically

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Purnachander-Konda/DeepfakeDetectionUsingSWINTransformer/blob/main/train_on_colab.ipynb)

### Train Locally (Advanced)

Requires an NVIDIA GPU and the FaceForensics++ dataset:

```bash
python swin-tiny-complete-training.py
```

**Training Configuration:**
- **Base Model**: `microsoft/swin-tiny-patch4-window7-224` (pretrained on ImageNet-1K)
- **Learning Rate**: 2e-5
- **Batch Size**: 16 (with gradient accumulation of 2)
- **Epochs**: 3
- **Optimizer**: AdamW with weight decay 0.01
- **Strategy**: Gradient checkpointing + FP16 mixed precision

---

## 📊 Results

<!-- TODO: Fill in your actual metrics after training -->

| Metric | Score |
|---|---|
| **Accuracy** | _Your result here_ |
| **F1 Score (Macro)** | _Your result here_ |
| **Precision (Macro)** | _Your result here_ |
| **Recall (Macro)** | _Your result here_ |

> Update these values with your actual training results from `results/swin-tiny-complete/`.

---

## 🌐 Live Demo

<!-- TODO: Update with your actual Hugging Face Spaces URL after deployment -->

The model is deployed on **Hugging Face Spaces** with a Gradio interface:

🔗 **[Try it here → huggingface.co/spaces/Purnachander-Konda/deepfake-detection-swin](https://huggingface.co/spaces/Purnachander-Konda/deepfake-detection-swin)**

---

## 🛠️ Tech Stack

- **Model**: [SWIN Transformer](https://arxiv.org/abs/2103.14030) (Tiny variant)
- **Framework**: [PyTorch](https://pytorch.org/) + [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- **Web UI**: [Gradio](https://gradio.app/)
- **Dataset**: [Deepfake and Real Images](https://huggingface.co/datasets/Hemg/deepfake-and-real-images)
- **Metrics**: HuggingFace Evaluate (Accuracy, F1, Precision, Recall)
- **Hosting**: [Hugging Face Spaces](https://huggingface.co/spaces)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Microsoft Research — SWIN Transformer](https://github.com/microsoft/Swin-Transformer)
- [Hemg/deepfake-and-real-images](https://huggingface.co/datasets/Hemg/deepfake-and-real-images) dataset
- [HuggingFace](https://huggingface.co/) for model hosting and Transformers library
- [Gradio](https://gradio.app/) for the web interface framework
- [Google Colab](https://colab.research.google.com/) for free GPU training

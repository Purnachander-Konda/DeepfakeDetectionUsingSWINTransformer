# 🔍 Deepfake Detection Using SWIN Transformer

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-UI-F97316?logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning-based system for detecting deepfake images using a fine-tuned **SWIN Transformer** (Shifted Window Transformer). The model classifies facial images as **Real** or **Fake** with confidence scores and is deployed as a live web application.

> **B.Tech Major Project** by Purna Chandar Konda

🚀 **[Try the Live Demo →](https://huggingface.co/spaces/Purnachander-Konda/deepfake-detection-swin)**

🤗 **[View Trained Model on HuggingFace Hub →](https://huggingface.co/Purnachander-Konda/deepfake-detection-swin)**

---

## 📋 Table of Contents

- [About](#-about)
- [How It Works](#-how-it-works)
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

The rapid advancement of deepfake technology has made it increasingly difficult to distinguish real facial images from manipulated ones, posing a serious threat to digital media integrity, cybersecurity, and trust in online content.

This project addresses the problem by leveraging the **SWIN Transformer** architecture — a hierarchical vision transformer that uses shifted window-based self-attention — to detect and classify manipulated facial images with high accuracy.

**Key highlights:**
- Fine-tuned **SWIN-Tiny** model (`microsoft/swin-tiny-patch4-window7-224`) pretrained on ImageNet-1K
- Trained on **190,000+** real and fake face images from the [Deepfake and Real Images](https://huggingface.co/datasets/Hemg/deepfake-and-real-images) dataset
- Deployed as a live **Gradio** web application on Hugging Face Spaces
- Includes a one-click **Google Colab** training notebook for easy reproducibility

---

## ⚙️ How It Works

```
┌─────────────┐    ┌──────────────────┐    ┌────────────────┐    ┌──────────────┐
│  Input Image │───▶│  Preprocessing    │───▶│ SWIN Transformer│───▶│ Classification│
│  (224×224)   │    │  (Resize, Norm)   │    │ (Feature Ext.)  │    │    Output    │
└─────────────┘    └──────────────────┘    └────────────────┘    └──────────────┘
                                                                        │
                                                                        ▼
                                                              ┌──────────────────┐
                                                              │  ✅ Real / ❌ Fake │
                                                              │  + Confidence %   │
                                                              └──────────────────┘
```

**Pipeline:**
1. **Input** — A facial image is uploaded through the web interface (any resolution).
2. **Preprocessing** — The image is resized to 224×224 pixels and normalized using ImageNet mean/std values.
3. **Feature Extraction** — The SWIN-Tiny transformer processes the image through 4 hierarchical stages using shifted window multi-head self-attention, extracting both local and global features.
4. **Classification** — A linear classification head maps the extracted 768-dimensional feature vector to 2 output classes (Real / Fake) using softmax probabilities.

---

## 📁 Project Structure

```
DeepfakeDetectionUsingSWINTransformer/
├── app.py                          # Gradio web app (deployed on HF Spaces)
├── demo.py                         # Quick local demo using HF pipeline
├── deploy_to_spaces.py             # One-click deployment script for HF Spaces
├── train_on_colab.ipynb            # Google Colab training notebook (recommended)
├── swin-tiny-complete-training.py  # Local training script (requires GPU)
├── model-testing.py                # Model evaluation script
├── image_extractor.py              # Frame extraction from video datasets
├── models/
│   └── swin-tiny-complete/         # Model configuration files
│       ├── config.json
│       └── preprocessor_config.json
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

> **Note:** Model weights (~110 MB) are hosted on [Hugging Face Hub](https://huggingface.co/Purnachander-Konda/deepfake-detection-swin) and are automatically downloaded when you run the app.

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Git
- (Optional) NVIDIA GPU with CUDA for local training

### Setup

```bash
# Clone the repository
git clone https://github.com/Purnachander-Konda/DeepfakeDetectionUsingSWINTransformer.git
cd DeepfakeDetectionUsingSWINTransformer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Model Weights

The trained model weights are hosted on [Hugging Face Hub](https://huggingface.co/Purnachander-Konda/deepfake-detection-swin) and are **downloaded automatically** when you run `app.py`. For manual download:

```bash
pip install huggingface-hub
huggingface-cli download Purnachander-Konda/deepfake-detection-swin --local-dir ./models/swin-tiny-complete
```

---

## 💻 Usage

### Run the Web App

```bash
python app.py
```

Open `http://localhost:7860` in your browser — upload any face image to get a Real/Fake prediction with confidence scores.

### Quick Demo

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

The fastest way to train — no local GPU needed:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Purnachander-Konda/DeepfakeDetectionUsingSWINTransformer/blob/main/train_on_colab.ipynb)

1. Open the notebook in Colab
2. Set **Runtime → GPU (T4)**
3. Run all cells — the notebook handles everything:
   - Downloads the 190K+ image dataset
   - Fine-tunes SWIN-Tiny for 3 epochs (~30-60 min)
   - Evaluates and prints metrics
   - Uploads the trained model to Hugging Face Hub

### Train Locally

Requires an NVIDIA GPU and the dataset:

```bash
python swin-tiny-complete-training.py
```

### Training Configuration

| Parameter | Value |
|---|---|
| **Base Model** | `microsoft/swin-tiny-patch4-window7-224` |
| **Dataset** | [Hemg/deepfake-and-real-images](https://huggingface.co/datasets/Hemg/deepfake-and-real-images) (190K+ images) |
| **Train/Test Split** | 80/20 (stratified) |
| **Learning Rate** | 2e-5 |
| **Batch Size** | 16 (×2 gradient accumulation = effective 32) |
| **Epochs** | 3 |
| **Optimizer** | AdamW (weight decay: 0.01) |
| **Precision** | FP16 mixed precision |
| **Checkpointing** | Gradient checkpointing enabled |
| **Total Parameters** | ~27.5M |

---

## 📊 Results

Evaluation metrics on the held-out test set (20% of dataset, ~38K images):

| Metric | Score |
|---|---|
| **Accuracy** | *0.9881* |
| **F1 Score (Macro)** | *0.9881* |
| **Precision (Macro)** | *0.9881* |
| **Recall (Macro)** | *0.9881* |

> The model achieves strong performance on binary deepfake classification. Metrics were computed using HuggingFace Evaluate on the stratified test split.

---

## 🌐 Live Demo

The model is deployed as a **Gradio** web application on **Hugging Face Spaces**:

🔗 **[https://huggingface.co/spaces/Purnachander-Konda/deepfake-detection-swin](https://huggingface.co/spaces/Purnachander-Konda/deepfake-detection-swin)**

Features:
- Upload any face image for instant Real/Fake classification
- Confidence scores for both classes
- No sign-up or installation required

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Model** | [SWIN Transformer](https://arxiv.org/abs/2103.14030) — Tiny variant |
| **Framework** | [PyTorch](https://pytorch.org/) + [HuggingFace Transformers](https://huggingface.co/docs/transformers) |
| **Web Interface** | [Gradio](https://gradio.app/) |
| **Dataset** | [Deepfake and Real Images](https://huggingface.co/datasets/Hemg/deepfake-and-real-images) — 190K+ images |
| **Evaluation** | HuggingFace Evaluate (Accuracy, F1, Precision, Recall) |
| **Training** | [Google Colab](https://colab.research.google.com/) — T4 GPU |
| **Model Hosting** | [Hugging Face Hub](https://huggingface.co/Purnachander-Konda/deepfake-detection-swin) |
| **App Hosting** | [Hugging Face Spaces](https://huggingface.co/spaces/Purnachander-Konda/deepfake-detection-swin) |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Microsoft Research](https://github.com/microsoft/Swin-Transformer) for the SWIN Transformer architecture
- [Hemg](https://huggingface.co/datasets/Hemg/deepfake-and-real-images) for the Deepfake and Real Images dataset
- [Hugging Face](https://huggingface.co/) for Transformers library, model hosting, and Spaces
- [Gradio](https://gradio.app/) for the web interface framework
- [Google Colab](https://colab.research.google.com/) for free GPU access

---

*Built by [Purna Chandar Konda](https://github.com/Purnachander-Konda)*

# 🔍 Deepfake Detection Using SWIN Transformer

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-UI-F97316?logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning system for detecting deepfake images using a fine-tuned **SWIN Transformer** (Shifted Window Transformer). The model classifies facial images as **Real** or identifies the specific **deepfake technique** used to generate them.

> **B.Tech Major Project** — Built with PyTorch, HuggingFace Transformers, and Gradio.

<!-- TODO: Uncomment after deploying to Hugging Face Spaces -->
<!-- 🚀 **[Try the Live Demo →](https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detection-swin)** -->

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

The model is fine-tuned on the [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset, which contains both real and manipulated face images generated using four different deepfake techniques.

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
                                                              │  + Fake Type     │
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
| **Deepfakes** | Face generated using autoencoder-based deepfake techniques |
| **Face2Face** | Facial reenactment — expressions transferred from source to target |
| **FaceSwap** | Identity swap — face region replaced with another person's face |
| **NeuralTextures** | Facial manipulation using learned neural textures |

---

## 📁 Project Structure

```
DeepfakeDetectionUsingSWINTransformer/
├── app.py                          # Gradio web app (for deployment)
├── demo.py                         # Local demo using HuggingFace pipeline
├── swin-tiny-complete-training.py   # Model training script
├── model-testing.py                 # Model evaluation script
├── image_extractor.py               # Frame extraction from FaceForensics++ videos
├── models/
│   └── swin-tiny-complete/          # Fine-tuned model files
│       ├── config.json
│       ├── preprocessor_config.json
│       └── pytorch_model.bin        # Model weights (not in repo — see Setup)
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
git clone https://github.com/YOUR_USERNAME/DeepfakeDetectionUsingSWINTransformer.git
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
huggingface-cli download YOUR_USERNAME/deepfake-detection-swin --local-dir ./models/swin-tiny-complete
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

### Dataset Preparation

1. Download the [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset.
2. Place videos in the `dataset/` directory following this structure:
   ```
   dataset/
   ├── original_sequences/youtube/c23/videos/
   └── manipulated_sequences/c23/videos{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/
   ```
3. Extract frames and split into train/test:
   ```bash
   python image_extractor.py
   ```

### Train the Model

```bash
python swin-tiny-complete-training.py
```

**Training Configuration:**
- **Base Model**: `microsoft/swin-tiny-patch4-window7-224` (pretrained on ImageNet-1K)
- **Learning Rate**: 2e-5
- **Batch Size**: 4 (with gradient accumulation of 4)
- **Epochs**: Configurable (default: 1 in script, increase for better results)
- **Optimizer**: AdamW with weight decay 0.01
- **Strategy**: Gradient checkpointing enabled for memory efficiency

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

🔗 **[Try it here → huggingface.co/spaces/YOUR_USERNAME/deepfake-detection-swin](https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detection-swin)**

---

## 🛠️ Tech Stack

- **Model**: [SWIN Transformer](https://arxiv.org/abs/2103.14030) (Tiny variant)
- **Framework**: [PyTorch](https://pytorch.org/) + [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- **Web UI**: [Gradio](https://gradio.app/)
- **Dataset**: [FaceForensics++](https://github.com/ondyari/FaceForensics)
- **Metrics**: HuggingFace Evaluate (Accuracy, F1, Precision, Recall)
- **Hosting**: [Hugging Face Spaces](https://huggingface.co/spaces)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Microsoft Research — SWIN Transformer](https://github.com/microsoft/Swin-Transformer)
- [FaceForensics++ Benchmark](https://github.com/ondyari/FaceForensics)
- [HuggingFace](https://huggingface.co/) for model hosting and Transformers library
- [Gradio](https://gradio.app/) for the web interface framework

"""
Deepfake Detection using SWIN Transformer
A Gradio web application for detecting deepfake images using a fine-tuned SWIN Transformer model.
"""

import gradio as gr
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# ── Configuration ──────────────────────────────────────────────────────────────
# Replace YOUR_USERNAME with your Hugging Face username after uploading the model
MODEL_ID = "YOUR_USERNAME/deepfake-detection-swin"  # TODO: Update this
LOCAL_MODEL_PATH = "./models/swin-tiny-complete"

# ── Load Model ─────────────────────────────────────────────────────────────────
try:
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    print(f"✓ Model loaded from Hugging Face Hub: {MODEL_ID}")
except Exception:
    processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH)
    print(f"✓ Model loaded from local path: {LOCAL_MODEL_PATH}")

model.eval()

# Label descriptions for user-friendly output
LABEL_INFO = {
    "real": ("✅ Real", "This image appears to be authentic."),
    "Deepfakes": ("❌ Deepfake", "This image was likely generated using Deepfakes."),
    "Face2Face": ("❌ Face2Face", "This image was likely manipulated using Face2Face reenactment."),
    "FaceSwap": ("❌ FaceSwap", "This image was likely created using FaceSwap technology."),
    "NeuralTextures": ("❌ NeuralTextures", "This image was likely manipulated using Neural Textures."),
}


def classify_image(image: Image.Image):
    """Classify an image as real or deepfake with confidence scores."""
    if image is None:
        return {}, "Please upload an image to analyze."

    # Preprocess and predict
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    # Build results
    scores = {}
    for idx, prob in enumerate(probabilities):
        label = model.config.id2label[str(idx)]
        display_label = LABEL_INFO.get(label, (label, ""))[0]
        scores[display_label] = float(prob)

    # Top prediction
    top_idx = probabilities.argmax().item()
    top_label = model.config.id2label[str(top_idx)]
    top_conf = probabilities[top_idx].item() * 100
    _, description = LABEL_INFO.get(top_label, (top_label, ""))

    verdict = f"**Prediction:** {LABEL_INFO.get(top_label, (top_label,))[0]}  \n"
    verdict += f"**Confidence:** {top_conf:.1f}%  \n"
    verdict += f"**Details:** {description}"

    return scores, verdict


# ── Gradio Interface ───────────────────────────────────────────────────────────
TITLE = "🔍 Deepfake Detection using SWIN Transformer"

DESCRIPTION = """
Upload a face image to detect whether it is **real** or a **deepfake**.

This model uses a fine-tuned **SWIN Transformer** (Shifted Window Transformer) to classify facial images into one of five categories:
**Real**, **Deepfakes**, **Face2Face**, **FaceSwap**, or **NeuralTextures**.

Built as a B.Tech Major Project — trained on the [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset.
"""

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Classification Scores"),
        gr.Markdown(label="Verdict"),
    ],
    title=TITLE,
    description=DESCRIPTION,
    article="**Model:** SWIN-Tiny (patch4-window7-224) fine-tuned for deepfake detection | **Framework:** PyTorch + HuggingFace Transformers",
    allow_flagging="never",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()

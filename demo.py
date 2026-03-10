import gradio as gr
from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor

pipe = pipeline("image-classification", "./models/swin-tiny-complete")
iface = gr.Interface.from_pipeline(pipe, allow_flagging=False)

iface.launch()

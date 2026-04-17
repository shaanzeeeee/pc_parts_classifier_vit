import json
from pathlib import Path

import gradio as gr
import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

MODEL_REPO_ID = "shaanzeeeee/vit_base_patch16_pc_parts_classifier"
MODEL_STATE_FILENAME = "best_model_state_dict.pth"
MODEL_META_FILENAME = "best_model_metadata.json"
MODEL_NAME = "vit_base_patch16_224"


def load_model_and_metadata():
    metadata_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_META_FILENAME,
        repo_type="model",
    )
    state_dict_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_STATE_FILENAME,
        repo_type="model",
    )

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    state_dict = torch.load(state_dict_path, map_location="cpu")

    backbone_state = {
        k.replace("0.model.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("0.model.")
    }
    head_state = {
        k.replace("1.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("1.")
    }

    backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
    backbone.load_state_dict(backbone_state, strict=True)

    head = nn.Sequential(
        nn.BatchNorm1d(768),
        nn.Dropout(p=0.25),
        nn.Linear(768, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.5),
        nn.Linear(512, metadata["num_classes"], bias=False),
    )
    head.load_state_dict(head_state, strict=True)

    model = nn.Sequential(backbone, head)
    model.eval()
    return model, metadata


model, metadata = load_model_and_metadata()
labels = metadata["classes"]
img_size = int(metadata["img_size"])

preprocess = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def predict(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    with torch.no_grad():
        logits = model(preprocess(image).unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    return {labels[i]: float(probs[i]) for i in range(len(labels))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload PC image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="PC Parts Classifier (ViT-Base)",
    description="Upload a PC component image to classify it into 1 of 11 classes.",
)

if __name__ == "__main__":
    demo.launch()

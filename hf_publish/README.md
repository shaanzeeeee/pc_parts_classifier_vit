---
language: en
license: mit
library_name: fastai
pipeline_tag: image-classification
tags:
- fastai
- timm
- vision-transformer
- image-classification
- pc-parts
widget:
- src: https://images.unsplash.com/photo-1587202372775-a457f4ad61b9
  example_title: PC build example
---

# vit_base_patch16_pc_parts_classifier

Vision Transformer image classifier for 11 PC component and cable-management classes.

## Model Details

- Architecture: ViT-Base Patch16 224 (`vit_base_patch16_224`)
- Framework: FastAI + timm + PyTorch
- Input size: 224x224 RGB
- Classes: 11
- Epochs: 15
- Batch size: 16
- Test accuracy: 0.7389
- Training date: 2026-04-17

## Labels

1. AIO_Liquid_Cooler
2. Air_Cooler
3. Bad_Cable_Management
4. CPU
5. Good_Cable_Management
6. Graphics_Card
7. M2_NVMe_Drive
8. Motherboard
9. PC_Case
10. Power_Supply
11. RAM_Stick

## Files

- `best_model_export.pkl`: FastAI export for direct inference.
- `best_model_state_dict.pth`: PyTorch state dict.
- `best_model_metadata.json`: Training and class metadata.

## Inference (FastAI)

```python
from fastai.learner import load_learner
from pathlib import Path

learn = load_learner("best_model_export.pkl")
pred_class, pred_idx, probs = learn.predict(Path("sample.jpg"))
print(pred_class)
print({learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))})
```

## Live Demo

A live inference demo is available on Hugging Face Spaces:

- https://huggingface.co/spaces/shaanzeeeee/vit-base-pc-parts-inference

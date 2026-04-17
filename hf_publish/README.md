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

## Training Procedure (15 Epochs)

The model was trained for a total of **15 epochs** using a fixed, reproducible split and transfer learning.

1. **Prepare dataset**
  - Input images are cleaned (duplicate filtering + integrity checks).
  - A fixed stratified split is used for train/validation/test.
2. **Build dataloaders**
  - Input resolution: `224x224` RGB.
  - Batch size: `16`.
  - Standard FastAI/timm image normalization and augmentation are applied.
3. **Initialize model**
  - Backbone: `vit_base_patch16_224` (pretrained).
  - Head is adapted for 11 target classes.
4. **Train for 15 epochs**
  - Fine-tune with FastAI training loop.
  - Track `train_loss`, `valid_loss`, and `accuracy` every epoch.
5. **Evaluate and export**
  - Evaluate on held-out test split.
  - Export FastAI learner + PyTorch state dict + metadata.

### ViT Training Log (Exact)

- Training model: `vit_base_patch16_224`
- Using base LR for `vit_base_patch16_224`: `8.32e-04`

Phase 1 (initial run output)

| epoch | train_loss | valid_loss | accuracy | time |
| ---: | ---: | ---: | ---: | --- |
| 0 | 2.275931 | 1.350919 | 0.566502 | 01:24 |

Phase 2 (15-epoch fine-tuning log)

| epoch | train_loss | valid_loss | accuracy | time |
| ---: | ---: | ---: | ---: | --- |
| 0 | 1.724283 | 1.237632 | 0.633005 | 01:34 |
| 1 | 1.589701 | 1.166143 | 0.628079 | 01:35 |
| 2 | 1.413170 | 1.055101 | 0.667488 | 01:35 |
| 3 | 1.222731 | 1.061895 | 0.674877 | 01:38 |
| 4 | 1.024032 | 1.045381 | 0.699507 | 01:42 |
| 5 | 0.898932 | 1.004452 | 0.709360 | 01:38 |
| 6 | 0.773990 | 0.993074 | 0.719212 | 01:38 |
| 7 | 0.643601 | 0.999479 | 0.714286 | 01:38 |
| 8 | 0.591031 | 1.007470 | 0.719212 | 01:39 |
| 9 | 0.522374 | 1.002515 | 0.711823 | 01:39 |
| 10 | 0.412975 | 1.012987 | 0.716749 | 01:38 |
| 11 | 0.382450 | 1.017962 | 0.714286 | 01:38 |
| 12 | 0.390728 | 1.024924 | 0.724138 | 01:37 |
| 13 | 0.335873 | 1.039688 | 0.719212 | 01:37 |
| 14 | 0.338043 | 1.024603 | 0.716749 | 01:37 |

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

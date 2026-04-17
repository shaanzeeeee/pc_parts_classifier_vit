# PC Components and Cable-Management Image Classifier

End-to-end computer vision pipeline for classifying PC hardware and cable-management quality into 11 categories using FastAI, timm, and PyTorch.

This repository is structured like a production-style ML project: data cleaning, fixed split strategy, multi-model benchmarking, held-out evaluation, artifact export, and Hugging Face deployment.

## At a Glance

| Item | Value |
| --- | --- |
| Task | Multi-class image classification |
| Number of classes | 11 |
| Dataset size (raw -> cleaned) | 4,249 -> 4,055 |
| Data quality controls | Corrupt-image filtering + SHA1 duplicate removal (including cross-class duplicates) |
| Models compared | ResNet50, EfficientNet-B3, ViT-Base Patch16 224 |
| Best model | ViT-Base Patch16 224 |
| Held-out test accuracy | 73.89% |
| Deployment | Hugging Face Model + Gradio Space |

## Links

- Hugging Face Model: https://huggingface.co/shaanzeeeee/vit_base_patch16_pc_parts_classifier
- Hugging Face Space (Inference Demo): https://huggingface.co/spaces/shaanzeeeee/vit-base-pc-parts-inference
- Hugging Face API (Space config): https://shaanzeeeee-vit-base-pc-parts-inference.hf.space/config
- Dataset: https://huggingface.co/datasets/shaanzeeeee/pc_parts_classifier_dataset
- Live Web App (GitHub Pages): https://shaanzeeeee.github.io/pc_parts_classifier_vit/

## Quick Navigation

- [Business / Product Value](#business--product-value)
- [Classes](#classes)
- [Pipeline](#pipeline)
- [Model Comparison](#model-comparison)
- [Why ViT-Base Won](#why-vit-base-won)
- [Confusion Analysis](#confusion-analysis)
- [Final Test Results](#final-test-results)
- [Artifacts and Deployment](#artifacts-and-deployment)
- [Reproducibility](#reproducibility)
- [Next Improvements](#next-improvements)

## Business / Product Value

Potential applications:

1. Automated cataloging/tagging of PC component images.
2. Quality-assurance checks for PC build photos (good vs bad cable management).
3. Moderation/recommendation workflows in e-commerce and enthusiast communities.

## Classes

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

## Pipeline

### 1) Data cleaning and indexing

- Input directory: `raw/`
- Cleaned output: `cleaned/`
- Corrupt images removed via verification.
- Duplicates removed using `sha1(file_bytes)` fingerprinting.
- Cross-class duplicates explicitly tracked and excluded.
- Cleaning metadata written to:
  - `cleaned/cleaning_manifest.json`
  - `cleaned/cleaned_index.csv`

Cleaning summary (from manifest):

- Raw images: 4,249
- Kept images: 4,055
- Removed corrupt: 0
- Removed duplicates: 194

### 2) Fixed reproducible split

- Shared fixed stratified split reused across all models.
- Train: 3,243
- Validation: 406
- Test: 406
- Split file: `cleaned/splits.csv`

### 3) Training setup

- Framework: FastAI + PyTorch + timm
- Input size: 224x224
- Epochs: 15
- Batch size: 16 (auto-selected for 8 GB GPU profile)
- Seed controls: FastAI, NumPy, Python `random`, and PyTorch
- Augmentation: Moderate geometry + stronger lighting jitter for product-style images

## Model Comparison

All models are trained under the same split and training conditions for fair comparison:

- ResNet50
- EfficientNet-B3
- ViT-Base Patch16 224

### Comparison methodology

Tracked per model:

- Training loss trajectory (`train_loss` per epoch)
- Validation loss trajectory (`valid_loss` per epoch)
- Validation accuracy trajectory (`accuracy` per epoch)
- Learning rate used (`lr_used`)

Ranking logic in notebook (`Model Comparison and Best Model Selection`):

1. Max validation accuracy (`best_val_accuracy`) - higher is better.
2. Final validation loss (`final_val_loss`) - lower is better.
3. Validation-loss stability (`val_loss_delta_std`) - lower is better.

### Side-by-side results table

| Model | Epochs | Validation Loss | Validation Accuracy | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Test Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vit_base_patch16_224 | 15 | 1.0244 | 0.7167 | 0.7389 | 0.7404 | 0.7333 | 0.7300 | 0.7345 |
| resnet50 | 15 | 1.1468 | 0.6700 | 0.7044 | 0.7017 | 0.6999 | 0.6984 | 0.7024 |
| efficientnet_b3 | 15 | 1.1546 | 0.6379 | 0.6010 | 0.5977 | 0.5997 | 0.5943 | 0.5972 |

## Why ViT-Base Won

Selected best model: `vit_base_patch16_224` (see [artifacts/best_model_metadata.json](artifacts/best_model_metadata.json)).

Key reasons:

1. Won the notebook's multi-criterion validation ranking (accuracy + loss + stability).
2. Generalized best on held-out test split (73.89% accuracy).
3. Strong separability on multiple hardware categories (for example RAM_Stick and PC_Case).
4. Transformer backbone aligned well with dataset diversity (closeups + full-build scenes).

Repository note: artifacts exist for all three model families in [artifacts](artifacts); deployed model is ViT.

## Confusion Analysis

Top off-diagonal confusion pairs on the fixed held-out test split (`true -> predicted`):

| Rank | ViT-Base confusion | Count | ResNet50 confusion | Count | EfficientNet-B3 confusion | Count |
| --- | --- | ---: | --- | ---: | --- | ---: |
| 1 | AIO_Liquid_Cooler -> Good_Cable_Management | 5 | Good_Cable_Management -> Bad_Cable_Management | 6 | AIO_Liquid_Cooler -> Bad_Cable_Management | 6 |
| 2 | Bad_Cable_Management -> Good_Cable_Management | 5 | AIO_Liquid_Cooler -> Good_Cable_Management | 5 | Good_Cable_Management -> Bad_Cable_Management | 6 |
| 3 | CPU -> Motherboard | 5 | AIO_Liquid_Cooler -> PC_Case | 4 | Air_Cooler -> AIO_Liquid_Cooler | 5 |
| 4 | Good_Cable_Management -> PC_Case | 4 | CPU -> Graphics_Card | 4 | Graphics_Card -> RAM_Stick | 5 |
| 5 | M2_NVMe_Drive -> Graphics_Card | 4 | Graphics_Card -> CPU | 4 | Power_Supply -> Bad_Cable_Management | 5 |

Interpretation:

- ViT-Base shows lower aggregate confusion and better macro/weighted F1 than CNN baselines.
- Most confusion appears between visually similar classes (for example cable-management pair, CPU vs Motherboard, cooling categories).
- This matches per-class variation in `reports/best_model_class_metrics.csv`.

## Final Test Results

Best model: `vit_base_patch16_224`

- Accuracy: 0.7389
- Macro F1: 0.7300
- Weighted F1: 0.7345

Per-class highlights:

- Strong classes:
  - RAM_Stick (F1: 0.9000)
  - PC_Case (F1: 0.8163)
  - Air_Cooler (F1: 0.8148)
  - CPU (F1: 0.8000)
- Harder classes:
  - AIO_Liquid_Cooler (F1: 0.6286)
  - Good_Cable_Management (F1: 0.6286)
  - Bad_Cable_Management (F1: 0.6333)

Saved reports:

- `reports/best_model_class_metrics.csv`
- `reports/best_model_classification_report.csv`

## Artifacts and Deployment

Model artifacts in `artifacts/`:

- `best_model_export.pkl` (FastAI learner export)
- `best_model_state_dict.pth` (PyTorch weights)
- `best_model_metadata.json` (classes, image size, metrics, timestamp)
- Optional ONNX export target in notebook (`best_model.onnx`)

Hugging Face assets:

- Model URL: https://huggingface.co/shaanzeeeee/vit_base_patch16_pc_parts_classifier
- Dataset URL: https://huggingface.co/datasets/shaanzeeeee/pc_parts_classifier_dataset
- Model card and files: `hf_publish/README.md`
- Space app: `hf_publish/space/app.py`
- Space requirements: `hf_publish/space/requirements.txt`
- Space URL: https://huggingface.co/spaces/shaanzeeeee/vit-base-pc-parts-inference
- Live frontend URL: https://shaanzeeeee.github.io/pc_parts_classifier_vit/

## Inference Utility

Notebook inference helper returns:

- predicted class
- predicted class index
- confidence scores for all 11 classes (sorted)

Implemented in the final section of `fastai_3_models.ipynb`.

## Project Structure

```text
.
|- fastai_3_models.ipynb
|- raw/
|- cleaned/
|  |- cleaning_manifest.json
|  |- cleaned_index.csv
|  |- splits.csv
|- reports/
|  |- best_model_class_metrics.csv
|  |- best_model_classification_report.csv
|- artifacts/
|  |- best_model_state_dict.pth
|  |- best_model_metadata.json
|- hf_publish/
|  |- README.md
|  |- space/
|     |- app.py
|     |- requirements.txt
|     |- README.md
|- index.html
|- styles.css
|- app.js
```

## Frontend (GitHub Pages UI)

A static frontend is included for simple public access via GitHub Pages, using the NVIDIA-inspired design system from `DESIGN.md`.

Frontend files:

- `index.html` - app shell and semantic layout
- `styles.css` - design tokens and responsive styling
- `app.js` - upload, inference request, and result rendering logic

Implemented features:

1. Drag-and-drop and click-upload image input
2. Local image preview
3. Top-5 prediction list with confidence bars
4. Class help/description panel
5. Copy and download prediction results
6. Loading and error states

Inference source:

- Public Hugging Face Space endpoint (`shaanzeeeee/vit-base-pc-parts-inference`)

Run locally (quick static preview):

```bash
python -m http.server 5500
```

Then open `http://localhost:5500`.

Deploy on GitHub Pages:

1. Push these frontend files to your repository root (already done in this repo layout).
2. In GitHub repo settings, enable Pages and select `main` branch with `/ (root)`.
3. Wait for Pages build and open the generated site URL.

## Reproducibility

### Environment

Python 3.10+ recommended.

Install dependencies:

```bash
pip install fastai timm torch torchvision scikit-learn seaborn matplotlib pandas numpy gradio huggingface_hub pillow
```

### Run training and evaluation

1. Open `fastai_3_models.ipynb`.
2. Run all cells from top to bottom.
3. Outputs are generated in `cleaned/`, `reports/`, and `artifacts/`.

## Engineering Notes

- Clean train/validation/test boundaries are enforced.
- Same split and transform policy reused across all models for fair benchmarking.
- Exported artifacts support both research reproducibility and deployment handoff.

## Next Improvements

1. Add class-balanced sampling or focal loss for harder cable-management classes.
2. Add test-time augmentation and confidence calibration.
3. Add lightweight variants for faster CPU/mobile inference.
4. Add drift checks and automated retraining hooks for new image distributions.

---

For recruiters/hiring managers: this repository demonstrates practical ML workflow depth across data quality, fair model selection, metrics interpretation, and deployment readiness.

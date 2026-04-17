# PC Components and Cable-Management Image Classifier

An end-to-end computer vision project that classifies PC hardware and cable-management quality into 11 categories using transfer learning with FastAI and timm.

This project was built as a production-style ML workflow, not just a notebook experiment: dataset cleaning, reproducible split strategy, multi-model benchmarking, held-out evaluation, model export, and Hugging Face deployment are all included.

## Recruiter Snapshot

- Problem type: Multi-class image classification (11 classes)
- Dataset scale: 4,249 raw images -> 4,055 cleaned images after de-duplication
- Data quality controls: Corrupt-image filtering + SHA1 duplicate removal (including cross-class duplicates)
- Modeling strategy: Fair comparison across 3 pretrained backbones (ResNet50, EfficientNet-B3, ViT-Base)
- Best model: ViT-Base Patch16 224
- Test accuracy (held-out split): 73.89%
- Deployment: Hugging Face Model + Gradio Space

## Business / Product Framing

This model can support:

1. Automated cataloging and tagging of PC part images.
2. Quality-assurance checks for PC build photos (good vs bad cable management).
3. Moderation or recommendation workflows in e-commerce, forums, and enthusiast apps.

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

## End-to-End Pipeline

### 1) Data cleaning and indexing

- Input source: `raw/`
- Output dataset: `cleaned/`
- Corrupt images removed with image verification.
- Duplicate images removed using `sha1(file_bytes)` fingerprints.
- Cross-class duplicates are explicitly tracked and excluded.
- Cleaning metadata is stored in:
	- `cleaned/cleaning_manifest.json`
	- `cleaned/cleaned_index.csv`

Cleaning summary (verified from manifest):

- Raw images: 4,249
- Kept images: 4,055
- Removed corrupt: 0
- Removed duplicates: 194

### 2) Reproducible split strategy

- Stratified split is fixed and reused for all models.
- Train: 3,243
- Validation: 406
- Test: 406
- Split file: `cleaned/splits.csv`

### 3) Training setup

- Framework: FastAI + PyTorch + timm
- Image size: 224x224
- Epochs: 15
- Batch size: Auto-selected based on GPU memory (8 GB profile selected 16)
- Seed control: Set for FastAI, NumPy, Python random, and PyTorch
- Data augmentation: Moderate geometry + stronger lighting jitter for product-style images

### 4) Model benchmarking

All models are trained under the same split and training conditions for a fair comparison:

- ResNet50
- EfficientNet-B3
- ViT-Base Patch16 224

Best model is selected based on validation behavior and then evaluated once on the untouched test split.

### 5) How the 3 models were compared (loss, accuracy, epochs)

To make model selection objective and reproducible, each model was trained on the same cleaned dataset, same split file, same image size, and same total epoch budget.

Common training setup used for comparison:

- Epochs: 15 per model
- Input size: 224x224
- Batch size: 16 (auto-selected for available 8 GB GPU profile)
- Train/valid/test split: shared fixed stratified split (`cleaned/splits.csv`)

Metrics tracked for each model:

- Training loss trajectory (`train_loss` per epoch)
- Validation loss trajectory (`valid_loss` per epoch)
- Validation accuracy trajectory (`accuracy` per epoch)
- Learning rate used (`lr_used`)

### Side-by-side training/results table

The table below summarizes the model comparison using the same fixed validation and test splits used in the notebook workflow.

| Model | Epochs | Validation Loss | Validation Accuracy | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Test Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vit_base_patch16_224 | 15 | 1.0244 | 0.7167 | 0.7389 | 0.7404 | 0.7333 | 0.7300 | 0.7345 |
| resnet50 | 15 | 1.1468 | 0.6700 | 0.7044 | 0.7017 | 0.6999 | 0.6984 | 0.7024 |
| efficientnet_b3 | 15 | 1.1546 | 0.6379 | 0.6010 | 0.5977 | 0.5997 | 0.5943 | 0.5972 |

Ranking logic implemented in the notebook (`Model Comparison and Best Model Selection`):

1. Max validation accuracy (`best_val_accuracy`) - higher is better.
2. Final validation loss (`final_val_loss`) - lower is better.
3. Validation-loss stability (`val_loss_delta_std`) - lower is better.

This means the selected model is not only accurate, but also relatively better-behaved in validation dynamics.

### 6) Why ViT-Base was selected as best

The best model selected by the comparison pipeline is `vit_base_patch16_224` (see [artifacts/best_model_metadata.json](artifacts/best_model_metadata.json)).

Why this is a reasonable winner in this project:

1. It won the notebook's multi-criterion validation ranking (accuracy + loss + stability).
2. It generalized to a solid held-out test accuracy of 73.89%.
3. The class-wise results show strong separability on multiple hardware categories (for example RAM_Stick and PC_Case), indicating useful feature learning beyond one or two easy classes.
4. ViT backbones often perform strongly on diverse visual patterns when enough augmentation and transfer learning are used, which aligns with this dataset's variety (component closeups and full-build cable-management scenes).

Note: the repository stores final model artifacts for all three families in [artifacts](artifacts), and the final deployed model is the selected ViT variant.

### 7) Confusion metrics snapshot (from fixed held-out test split)

To visualize model behavior beyond aggregate accuracy, below are the highest off-diagonal confusion pairs per model (true class -> predicted class).

| Rank | ViT-Base confusion | Count | ResNet50 confusion | Count | EfficientNet-B3 confusion | Count |
| --- | --- | ---: | --- | ---: | --- | ---: |
| 1 | AIO_Liquid_Cooler -> Good_Cable_Management | 5 | Good_Cable_Management -> Bad_Cable_Management | 6 | AIO_Liquid_Cooler -> Bad_Cable_Management | 6 |
| 2 | Bad_Cable_Management -> Good_Cable_Management | 5 | AIO_Liquid_Cooler -> Good_Cable_Management | 5 | Good_Cable_Management -> Bad_Cable_Management | 6 |
| 3 | CPU -> Motherboard | 5 | AIO_Liquid_Cooler -> PC_Case | 4 | Air_Cooler -> AIO_Liquid_Cooler | 5 |
| 4 | Good_Cable_Management -> PC_Case | 4 | CPU -> Graphics_Card | 4 | Graphics_Card -> RAM_Stick | 5 |
| 5 | M2_NVMe_Drive -> Graphics_Card | 4 | Graphics_Card -> CPU | 4 | Power_Supply -> Bad_Cable_Management | 5 |

Interpretation:

- ViT-Base has lower aggregate confusion and better macro/weighted F1 compared to the CNN baselines.
- Most confusion happens among visually similar categories (e.g., cable-management pair, CPU vs Motherboard, cooling categories).
- This aligns with class-level recall/F1 variation reported in `reports/best_model_class_metrics.csv`.

## Final Evaluation (Held-out Test)

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

## Artifacts and Deployment Assets

Model artifacts are exported to `artifacts/`:

- `best_model_export.pkl` (FastAI learner export)
- `best_model_state_dict.pth` (PyTorch weights)
- `best_model_metadata.json` (classes, image size, metrics, timestamp)
- Optional ONNX export path in notebook (`best_model.onnx`)

Hugging Face assets:

- Model card and files: `hf_publish/README.md`
- Space app: `hf_publish/space/app.py`
- Space requirements: `hf_publish/space/requirements.txt`
- Space URL: https://huggingface.co/spaces/shaanzeeeee/vit-base-pc-parts-inference

## Inference API (Notebook Utility)

The project includes a reusable inference helper that returns:

- predicted class
- predicted class index
- confidence for all 11 classes (sorted)

Implemented in the final notebook section.

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
	 |- README.md
	 |- space/
			|- app.py
			|- requirements.txt
			|- README.md
```

## Reproducibility

### Environment

Python 3.10+ recommended.

Install core dependencies:

```bash
pip install fastai timm torch torchvision scikit-learn seaborn matplotlib pandas numpy gradio huggingface_hub pillow
```

### Run training and evaluation

1. Open `fastai_3_models.ipynb`.
2. Run all cells from top to bottom.
3. Outputs are generated into `cleaned/`, `reports/`, and `artifacts/`.

## Engineering Notes

- The workflow enforces clean train/validation/test boundaries.
- The same split and transforms are reused across models for reliable comparisons.
- Exported artifacts support both research reproducibility and deployment handoff.

## Next Improvements

1. Add class-balanced sampling or focal loss for the harder cable-management classes.
2. Add test-time augmentation and confidence calibration.
3. Add lightweight model variants for faster inference on CPU/mobile.
4. Add drift checks and automated retraining hooks for new incoming image distributions.

---

If you are a recruiter or hiring manager, this repository demonstrates practical ML skills across data quality, model selection, metrics interpretation, and deployment, with reproducible outputs and clear artifact management.

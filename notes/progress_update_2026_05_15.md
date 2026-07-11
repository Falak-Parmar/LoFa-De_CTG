# Project Progress Update - May 2026

## Overview
The core machine learning pipeline for **LoFa-De_CTG** (Logical Fallacy Detection and Controlled Text Generation) is now fully implemented and validated. The project has transitioned from a data-exploration phase to a functional training and inference system.

## Latest Implementations

### 1. Model Architectures (`src/model.py`)
- **Detection:** Implemented `FallacyDetectionModel` using **DeBERTa-v3-base**. Configured for 9-class classification (8 fallacies + 'none').
- **Generation:** Implemented `FallacyGenerationModel` supporting Causal LM (**GPT-2**) and Seq2Seq (**T5/Flan-T5**) for controlled text generation.

### 2. Data Engineering (`src/data_loader.py`)
- Created `FallacyDataset` for classification, which tokenizes a combination of News Title, Parent Comment, and Target Comment.
- Created `FallacyGenerationDataset` for CTG, formatting inputs as: `Fallacy: <type> | Title: <title> | Parent: <parent> | Comment: <target>`.
- Automated Label Mapping to ensure consistent class indexing across training, validation, and testing.

### 3. Training & Evaluation (`src/trainer.py`)
- Developed a unified `Trainer` class that handles GPU/CPU acceleration, AdamW optimization, and progress tracking via `tqdm`.
- Included automated evaluation producing detailed **Classification Reports** (Precision, Recall, F1-score) and Macro-F1 tracking.
- Implemented model checkpointing to the `models/` directory.

### 4. Pipeline Orchestration (`main.py`)
- Refactored the entry point into a multi-stage CLI:
    - `preprocess`: Converts raw JSON data from CoCoLoFa to processed CSVs.
    - `detect`: Executes the fine-tuning of the detection model.
    - `generate`: Executes the training of the controlled generation model.
    - `evaluate`: Provides an interactive interface for real-time inference.

## Current Project State

| Component | Status | Notes |
| :--- | :--- | :--- |
| Preprocessing | **Complete** | Data cleaned and formatted into `data/processed/`. |
| Detection Model | **Functional** | Base weights cached; ready for full fine-tuning. |
| Generation Model | **Functional** | GPT-2 integrated; labels masked for prompt-target training. |
| Dependencies | **Updated** | `requirements.txt` includes all necessary ML and utility libraries. |

## How to Run

### Preprocess Data
```bash
python3 main.py --stage preprocess
```

### Train Detector
```bash
python3 main.py --stage detect --epochs 3 --batch_size 16
```

### Interactive Inference (Detection)
```bash
python3 main.py --stage evaluate --model microsoft/deberta-v3-base
```

### Interactive Inference (Generation)
```bash
python3 main.py --stage evaluate --model gpt2 --fallacy "slippery slope"
```

## Next Steps
1. **Full Training Run:** Perform a complete training cycle (3-5 epochs) on the full dataset to establish baseline performance.
2. **Hyperparameter Tuning:** Experiment with learning rates and batch sizes to optimize Macro-F1.
3. **Advanced Generation:** Implement prefix-tuning or P-tuning for more robust controlled generation.

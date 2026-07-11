# LoFa-De_CTG: Logical Fallacy Detection & Controlled Text Generation

This repository explores whether neural networks can truly detect logical fallacies and generate text that maintains logical consistency, highlighting the gap between language fluency and reasoning.

---

## 📈 Project Status & Milestones

### 🏁 Stage 1: Logical Fallacy Classification (COMPLETED)
*   **Model:** Fine-tuned `roberta-base` for 3 epochs with balanced class weighting.
*   **Performance:**
    *   **Test F1 Score:** **78.87%** (Val F1: 77.47%, Test Accuracy: 78.00%).
    *   **Baseline Improvement:** **+3.82%** absolute gain over the unweighted 1-epoch baseline (75.05%).
*   **Key Achievement:** Class weighting resolved the severe dataset imbalance, boosting minority class recall (e.g., `hasty generalization` recall rose from **30% ➔ 70%**).
*   **Robustness Probing:** Achieved **66.67% (8/12 passed)** on our counterfactual suite. The failures on valid inversions empirically document the gap between language fluency and reasoning.

### ⏳ Stage 2: Controlled Fallacy Generation (CTG) (READY)
*   The data loading and model configuration pipelines for both causal (GPT-2) and seq2seq (T5) architectures are fully integrated and ready.

---

## 📂 Project Structure

*   `data/`: Contains raw (`.json`) and processed (`.csv`) splits of the CoCoLoFa dataset.
*   `src/`:
    *   `preprocess.py`: Processes and cleans raw comments into training contexts.
    *   `data_loader.py`: Dataset loaders for classification (detection) and generation (CTG).
    *   `model.py`: Model wrappers for Sequence Classification and Causal/Seq2Seq language modeling.
    *   `trainer.py`: Custom training and evaluation logic with class-weighting and MPS fallback mechanisms.
*   `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and prototyping.
*   `notes/`: Markdown files detailing model logs, progress updates, and milestone reports.
*   `models/`: (Local only) Saved model checkpoints (`.pt` files).
*   `probe_counterfactuals.py`: Automated testing suite validating model robustness against synonym swaps and distractors.
*   `main.py`: Command-line entry point to orchestrate stages of the project.

---

## 🚀 How to Run

### 1. Installation
Install project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data
To clean the raw dataset and extract comments:
```bash
PYTHONPATH=. python3 main.py --stage preprocess
```

### 3. Train Fallacy Classifier (Stage 1)
To train the logical fallacy classification model:
```bash
PYTHONPATH=. python3 main.py --stage detect --model roberta-base --epochs 3 --batch_size 16
```
*Note: Due to PyTorch macOS MPS backend bugs with AdamW, training will automatically run stably on the CPU.*

### 4. Run Counterfactual Probing
To test a trained classification model against logical inversions and synonym swaps:
```bash
PYTHONPATH=. python3 probe_counterfactuals.py --model roberta-base
```

### 5. Classification Inference
Run inference interactively on custom text using a trained model:
```bash
PYTHONPATH=. python3 main.py --stage evaluate --model roberta-base
```
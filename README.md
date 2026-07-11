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

## 📝 Progress & Analysis Summary (Stage 1)

### 1. Empirical Verification
If the network were incapable of recognizing logical fallacies, performance on the 9-class dataset would be close to random noise (~11.1% accuracy). By fine-tuning the model using **Inverse-Frequency Class Weighting**, the classifier achieved a **78.87% Macro F1 score**. This confirms that the model has successfully mapped the statistical features characterizing different fallacy styles.

### 2. Semantic Generalization (Synonym Swaps)
To prove the model generalizes semantically rather than relying on exact word memorization, I used counterfactual Synonym Swap tests:
*   Replacing *"tradition"* with *"age-old custom"* and *"routinely practiced"* successfully preserved the correct `appeal to tradition` classification.
*   Replacing *"Dr. Smith"* with *"the most prominent expert in the field"* successfully preserved the `appeal to authority` classification.

### 3. Fluency vs. Reasoning Gap
The model's robustness suite failures provide empirical evidence of the gap between fluency and reasoning:
*   **Valid Inversions:** The model failed on structurally valid arguments that contained fallacy-associated words. For example, it misclassified *"Dr. Smith supported the theory with ten years of experimental data"* (a valid empirical citation) as `appeal to authority`.
*   The model recognizes superficial rhetorical structures and linguistic patterns, but lacks the formal reasoning logic to separate valid evidence from fallacious reasoning.

---

## 📂 Project Structure

*   `data/`: Contains raw (`.json`) and processed (`.csv`) splits of the CoCoLoFa dataset.
*   `src/`:
    *   `preprocess.py`: Processes and cleans raw comments into training contexts.
    *   `data_loader.py`: Dataset loaders for classification (detection) and generation (CTG).
    *   `model.py`: Model wrappers for Sequence Classification and Causal/Seq2Seq language modeling.
    *   `trainer.py`: Custom training and evaluation logic with class-weighting and MPS fallback mechanisms.
*   `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and prototyping.
*   `notes/`: (Ignored locally) Markdown files detailing model logs, progress updates, and milestone reports.
*   `models/`: (Ignored locally) Saved model checkpoints (`.pt` files).
*   `probe_counterfactuals.py`: Automated testing suite validating model robustness against synonym swaps and distractors.
*   `main.py`: Command-line entry point to orchestrate stages of the project.

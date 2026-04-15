# LoFa-De_CTG: Logical Fallacy Detection and Controlled Text Generation

Logical Fallacy Detection and Controlled Text Generation. A project exploring whether neural networks can detect logical fallacies and generate text that maintains logical consistency, highlighting the gap between language fluency and reasoning.

## Goals

The primary objective is to investigate the extent to which Neural Networks (NNs) truly understand logic versus mere linguistic patterns. We aim to:
1. **Detect:** Identify 8 common logical fallacies in human/LLM-assisted comments.
2. **Generate:** Controlled Text Generation (CTG) to produce arguments that either strictly adhere to logical consistency or intentionally employ specific fallacies.
3. **Analyze:** Evaluate the gap between a model's "fluency" and its "logical integrity."

## Project Structure

- `data/`: CoCoLoFa dataset splits.
- `src/`: 
  - `preprocess.py`: Data cleaning and context extraction.
  - `model.py`: Architectures for DeBERTa (Detection) and GPT-2/T5 (Generation).
  - `trainer.py`: Training loops for classification and causal language modeling.
- `exp.ipynb`: Data exploration and baseline testing.

## Dataset

Utilizes the **CoCoLoFa** dataset (EMNLP 2024), containing 7,706 comments across 8 fallacy categories:
- Appeal to Authority, Majority, Nature, Tradition, Worse Problems.
- False Dilemma, Hasty Generalization, Slippery Slope.
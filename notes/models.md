# Model Recommendations for LoFa-De_CTG

Based on an M4 MacBook Air (16GB RAM) and the dual goals of Fallacy Detection and Controlled Text Generation (CTG).

## 1. Detection (Encoder Models)
- **Primary Choice:** `microsoft/deberta-v3-base`
  - **Why:** Disentangled attention is superior for logical relationship modeling. It outperforms RoBERTa and BERT in NLU tasks.
  - **Local Impact:** ~130M parameters; very efficient on Apple Silicon (MPS).

## 2. Controlled Text Generation (Local)
- **Primary Choice:** `google/flan-t5-base`
  - **Why:** Instruction-tuned. Can be prompted for specific logic: *"Generate an argument for [X] using a [FALLACY] fallacy."*
- **Alternative:** `distilgpt2`
  - **Why:** Extremely lightweight if you want to use Control Tokens for style/logic steering.

## 3. Gold Standard / Data Augmentation (Cloud)
- **Primary Choice:** `Gemini 1.5 Flash` (Google AI Studio)
  - **Why:** Zero local RAM usage. Deep reasoning capabilities that far exceed small local models. Use it to generate high-quality synthetic data or labels.

## Strategy Summary
- Establish a zero-shot baseline using **Gemini 1.5 Flash**.
- Fine-tune **DeBERTa-v3-base** for high-speed, private detection.
- Use **Flan-T5** or **Gemini** for the CTG pipeline to explore the gap between fluency and reasoning.

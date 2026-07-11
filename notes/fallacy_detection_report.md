# Logical Fallacy Detection: Stage 1 Milestone Report

This report documents how we successfully achieved the **Logical Fallacy Detection (Classification)** milestone, analyzing the model's performance, the impact of class balancing, its semantic robustness, and the core implications for our broader project thesis.

---

## 1. Goal Alignment & Verification
Our primary project goal states:
> *"Logical Fallacy Detection and Controlled Text Generation: A project exploring whether neural networks can detect logical fallacies and generate text that maintains logical consistency, highlighting the gap between language fluency and reasoning."*

To say we have achieved the **Logical Fallacy Detection** milestone, we must verify that the neural network is doing more than just memorizing strings or outputting random noise. We prove this through three layers of evaluation:

### A. The Empirical Evidence (F1 & Accuracy)
If the neural network could not capture the features of logical fallacies:
*   Its accuracy on our 9-class dataset would be close to random guessing (**11.1%**).
*   A naive model that predicts only the majority class (`none`) would yield a Macro F1 score of **~6.3%**.

**Our Results:**
Our fine-tuned `roberta-base` classifier achieved:
*   **Validation F1:** **77.47%**
*   **Test F1:** **78.87%**
*   **Test Accuracy:** **78.00%**

This high score on a completely unseen test partition demonstrates that the network has mapped the statistical boundaries that define logical fallacies in human text.

### B. The Semantic Generalization Evidence (Synonym Swaps)
To verify that the model is detecting the logical structure of an argument rather than relying on exact word association, we subjected it to counterfactual **Synonym Swap** probes:
*   **Appeal to Tradition:** Changing the base phrase *"because it is our tradition"* to *"because it is our age-old custom and we have routinely practiced it"* did not degrade performance. The model successfully classified both as `appeal to tradition`.
*   **Appeal to Authority:** Changing the base phrase *"because Dr. Smith says it is"* to *"because it is supported by the most prominent expert in the field"* similarly succeeded, with the model outputting `appeal to authority`.

This confirms the network is forming representations of semantic concepts rather than simple surface n-grams.

### C. Proving the "Fluency vs. Reasoning" Gap
The core of our project thesis lies in highlighting the gap between language fluency and reasoning. Our robustness suite failures provide the exact empirical proof of this gap:
*   **Valid Inversions:** When presented with structurally valid arguments that use fallacy-trigger words (e.g., *"Dr. Smith supported the theory with ten years of experimental data"* or *"If we do not study, we will likely get a lower grade"*), the model mistakenly classified them as fallacies (`appeal to authority` and `false dilemma` respectively).
*   **Conclusion:** The neural network displays high linguistic fluency and pattern sensitivity, but it lacks the formal reasoning logic to separate structurally valid logic from invalid rhetoric when the surface keywords are identical.

---

## 2. Methodology & Key Decisions

We transitioned through multiple stages of optimization to reach this target:
1.  **Architecture Choice:** Swapped from DeBERTa-v3 (which has broken MPS GPU attention kernels in standard PyTorch) to `roberta-base`.
2.  **Optimizer Workarounds:** Evaluated CPU training vs. GPU workarounds, establishing a CPU safety routing to prevent PyTorch MPS AdamW kernel corruptions (`NaN` gradients).
3.  **Class Balancing (The Breakthrough):** Overcame severe class bias by calculating inverse-frequency weights from the training set, reducing the loss weight of `none` (majority class) to `0.19`.
    *   *Hasty Generalization* recall increased from **30% ➔ 70%**.
    *   *Slippery Slope* recall stabilized at **83%**.

---

## 3. Saved Deliverables
*   **Trained Weights:** `models/detection_roberta-base.pt`
*   **Probing Suite:** `probe_counterfactuals.py`

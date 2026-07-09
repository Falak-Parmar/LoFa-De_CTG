import argparse
import os
import torch
import pandas as pd
from transformers import AutoTokenizer
from src.model import FallacyDetectionModel

# Default configurations (can be overridden via CLI)
TRAIN_CSV = "data/processed/train_processed.csv"

def get_label_map(train_csv):
    df = pd.read_csv(train_csv)
    unique_labels = sorted(df['fallacy'].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    inv_label_map = {i: label for i, label in enumerate(unique_labels)}
    return label_map, inv_label_map

# Define the Counterfactual Suite
# Format: (text, expected_label, test_category, description)
test_suite = [
    # ------------------ APPEAL TO TRADITION ------------------
    (
        "We should maintain this system because it is our tradition.",
        "appeal to tradition",
        "Base Fallacy",
        "Standard tradition fallacy"
    ),
    (
        "We should maintain this system because it is our age-old custom and we have routinely practiced it.",
        "appeal to tradition",
        "Synonym Swap",
        "Replaced 'tradition' with synonyms"
    ),
    (
        "We studied the ancient tradition of candle-making in class, but we do not use it today.",
        "none",
        "Concept Distractor",
        "Contains word 'tradition' in a purely descriptive, non-argument context"
    ),
    
    # ------------------ APPEAL TO AUTHORITY ------------------
    (
        "This theory must be true because Dr. Smith says it is.",
        "appeal to authority",
        "Base Fallacy",
        "Standard appeal to authority"
    ),
    (
        "This theory must be true because it is supported by the most prominent expert in the field.",
        "appeal to authority",
        "Synonym Swap",
        "Replaced 'Dr. Smith' with 'prominent expert'"
    ),
    (
        "Dr. Smith explained the theory and supported it with ten years of experimental data.",
        "none",
        "Valid Inversion",
        "Citing authority with active empirical evidence"
    ),
    
    # ------------------ FALSE DILEMMA ------------------
    (
        "Either we ban cars completely, or the planet will die tomorrow.",
        "false dilemma",
        "Base Fallacy",
        "Standard false dilemma using 'Either/or'"
    ),
    (
        "We have only two options: outlaw vehicles entirely, or face total environmental collapse.",
        "false dilemma",
        "Synonym Swap",
        "Replaced 'Either/or' with 'only two options'"
    ),
    (
        "We can either go to the cinema or stay home to watch a movie; both sound nice to me.",
        "none",
        "Concept Distractor",
        "Non-extreme either/or choice, no fallacy"
    ),
    
    # ------------------ SLIPPERY SLOPE ------------------
    (
        "If you let students choose their seats, they will stop studying, fail their exams, and end up in prison.",
        "slippery slope",
        "Base Fallacy",
        "Standard slippery slope chain"
    ),
    (
        "Allowing students to select where they sit leads inevitably to academic decline, which causes dropout rates to skyrocket, ending in a life of crime.",
        "slippery slope",
        "Synonym Swap",
        "Replaced keywords with 'leads inevitably to' and 'skyrocket'"
    ),
    (
        "If we do not study for the exam, we will likely get a lower grade.",
        "none",
        "Valid Inversion",
        "Reasonable causal relationship without exaggeration"
    ),
]

def format_input(comment_text):
    # Formats text to match the combined context format the model was trained on
    return f"Title: General Discussion | Parent:  | Comment: {comment_text}"

def main():
    parser = argparse.ArgumentParser(description="Probe logical fallacy model counterfactuals")
    parser.add_argument("--model", type=str, default="roberta-base", help="Model name (e.g. roberta-base, microsoft/deberta-v3-base)")
    args = parser.parse_args()
    
    model_name = args.model
    model_path = f"models/detection_{model_name.split('/')[-1]}.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights not found at {model_path}")
        print(f"Please train the model first using: python3 main.py --stage detect --model {model_name} --epochs 1")
        return

    print("Loading label map...")
    label_map, inv_label_map = get_label_map(TRAIN_CSV)
    num_labels = len(label_map)

    print(f"Loading tokenizer and model for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FallacyDetectionModel(model_name, num_labels=num_labels)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    results = []
    correct_count = 0
    total_count = len(test_suite)
    
    print("\nRunning Counterfactual Tests...")
    print("-" * 80)
    
    for comment_text, expected_label, category, desc in test_suite:
        formatted = format_input(comment_text)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            pred_idx = torch.argmax(outputs.logits, dim=1).item()
            pred_label = inv_label_map[pred_idx]
            
        is_correct = (pred_label == expected_label)
        if is_correct:
            correct_count += 1
            
        results.append({
            "Text": comment_text,
            "Category": category,
            "Description": desc,
            "Expected": expected_label,
            "Predicted": pred_label,
            "Result": "PASS" if is_correct else "FAIL"
        })
        
    df_results = pd.DataFrame(results)
    
    # Print clean terminal report
    for idx, row in df_results.iterrows():
        print(f"[{row['Result']}] Category: {row['Category']} - {row['Description']}")
        print(f"  Text: \"{row['Text']}\"")
        print(f"  Expected: {row['Expected']} | Predicted: {row['Predicted']}\n")
        
    robustness_score = (correct_count / total_count) * 100
    print("-" * 80)
    print(f"Overall Robustness Score: {robustness_score:.2f}% ({correct_count}/{total_count} passed)")
    print("-" * 80)

    # Save to a Markdown Artifact
    artifact_path = "/Users/falak/.gemini/antigravity-cli/brain/5902fad9-cf89-43e4-9390-7f753ab54870/counterfactual_report.md"
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    
    with open(artifact_path, "w") as f:
        f.write("# Logical Fallacy Detection: Counterfactual Probing Report\n\n")
        f.write(f"This report assesses the robustness of the fine-tuned `{model_name}` model against lexical shortcuts and structural variations.\n\n")
        f.write(f"### Robustness Score: **{robustness_score:.2f}%** ({correct_count}/{total_count} passed)\n\n")
        f.write("## Detailed Test Results\n\n")
        f.write("| Result | Category | Description | Text | Expected | Predicted |\n")
        f.write("|---|---|---|---|---|---|\n")
        for idx, row in df_results.iterrows():
            res_str = "✅ PASS" if row['Result'] == "PASS" else "❌ FAIL"
            f.write(f"| {res_str} | {row['Category']} | {row['Description']} | \"{row['Text']}\" | `{row['Expected']}` | `{row['Predicted']}` |\n")
            
        f.write("\n\n## Analysis Guidelines\n")
        f.write("- **Base Fallacy (PASS) + Synonym Swap (FAIL):** Model is dependent on specific lexical keywords and lacks semantic generalization.\n")
        f.write("- **Concept Distractor (FAIL):** Model suffers from keyword triggers (false positives) even when no fallacious argument is made.\n")
        f.write("- **Valid Inversion (FAIL):** Model fails to understand the directionality and context of structural logic (e.g. confuses evidence citations with appeal to authority).\n")

    print(f"Detailed Markdown report saved to {artifact_path}")

if __name__ == "__main__":
    main()

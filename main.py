import argparse
import sys
import os
from src.preprocess import preprocess_cocolofa
from src.trainer import train_detection, train_generation

def main():
    parser = argparse.ArgumentParser(description="LoFa-De_CTG: Fallacy Detection & Controlled Generation")
    parser.add_argument("--stage", type=str, choices=["preprocess", "detect", "generate", "evaluate"], 
                        default="preprocess", help="Pipeline stage to run")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base", 
                        help="Model (e.g., microsoft/deberta-v3-base, gpt2, flan-t5-base)")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--fallacy", type=str, help="Target fallacy for CTG (generation stage)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. cpu, mps, cuda)")
    
    args = parser.parse_args()
    
    print(f"Running stage: {args.stage} using {args.model}")
    
    train_csv = "data/processed/train_processed.csv"
    dev_csv = "data/processed/dev_processed.csv"
    test_csv = "data/processed/test_processed.csv"
    
    if args.stage == "preprocess":
        raw_base = "data/raw"
        processed_base = "data/processed"
        os.makedirs(processed_base, exist_ok=True)
        for split in ["train", "dev", "test"]:
            input_file = f"{raw_base}/{split}.json"
            if os.path.exists(input_file):
                print(f"Processing {split}...")
                df = preprocess_cocolofa(input_file)
                df.to_csv(f"{processed_base}/{split}_processed.csv", index=False)
                print(f"Saved {split}_processed.csv with {len(df)} rows.")
            else:
                print(f"File not found: {input_file}")

    elif args.stage == "detect":
        train_detection(
            args.model, 
            train_csv, dev_csv, test_csv, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            device=args.device
        )

    elif args.stage == "generate":
        # For generation, if model is not specified, default to gpt2
        model_name = args.model if "gpt2" in args.model or "t5" in args.model else "gpt2"
        train_generation(
            model_name,
            train_csv, dev_csv, test_csv,
            epochs=args.epochs,
            batch_size=args.batch_size if args.batch_size <= 8 else 8, # GPT-2 is larger
            device=args.device
        )

    elif args.stage == "evaluate":
        from src.model import FallacyDetectionModel, FallacyGenerationModel
        from transformers import AutoTokenizer
        import torch

        if "deberta" in args.model:
            # Classification inference
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = FallacyDetectionModel(args.model, num_labels=9)
            model_path = f"models/detection_{args.model.split('/')[-1]}.pt"
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            
            sample_text = input("Enter text to analyze: ")
            inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
            
            # Map pred back to label (sorted alphabetically as in data_loader)
            import pandas as pd
            unique_labels = sorted(pd.read_csv(train_csv)['fallacy'].unique())
            print(f"Predicted Fallacy: {unique_labels[pred]}")

        elif "gpt2" in args.model or "t5" in args.model:
            # Generation inference
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = FallacyGenerationModel(args.model)
            model_path = f"models/generation_{args.model.split('/')[-1]}.pt"
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            
            fallacy = args.fallacy or "none"
            title = input("Enter news title: ")
            parent = input("Enter parent comment: ")
            
            prompt = f"Fallacy: {fallacy} | Title: {title} | Parent: {parent} | Comment: "
            inputs = tokenizer(prompt, return_tensors="pt")
            
            output_ids = model.generate(
                inputs["input_ids"], 
                attention_mask=inputs["attention_mask"],
                max_length=150,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print("-" * 30)
            print(f"Generated Argument:\n{generated_text.split('Comment: ')[-1]}")

if __name__ == "__main__":
    main()

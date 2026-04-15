import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="LoFaCa: Logical Fallacy Classification Pipeline")
    parser.add_argument("--stage", type=str, choices=["preprocess", "train", "evaluate"], 
                        default="preprocess", help="Pipeline stage to run")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Base model architecture")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    
    args = parser.parse_args()
    
    print(f"Running stage: {args.stage} with model: {args.model}")
    
    if args.stage == "preprocess":
        # TODO: Import and call src.preprocess
        pass
    elif args.stage == "train":
        # TODO: Import and call src.trainer
        pass
    elif args.stage == "evaluate":
        # TODO: Import and call src.trainer.evaluate
        pass

if __name__ == "__main__":
    main()

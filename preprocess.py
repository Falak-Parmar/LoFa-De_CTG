import json
import pandas as pd
from bs4 import BeautifulSoup
import os

def clean_html(text):
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def preprocess_cocolofa(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    rows = []
    for article in data:
        article_id = article["id"]
        title = article["title"]
        content_raw = article["content"]
        content_clean = clean_html(content_raw)
        
        # Map comments by ID for quick response context lookup
        comment_map = {c["id"]: c["comment"] for c in article["comments"]}
        
        for comment in article["comments"]:
            parent_id = comment.get("respond_to")
            parent_text = comment_map.get(parent_id, "") if parent_id else ""
            
            rows.append({
                "article_id": article_id,
                "news_title": title,
                "news_content": content_clean,
                "comment_id": comment["id"],
                "parent_comment": parent_text,
                "comment_text": comment["comment"],
                "fallacy": comment["fallacy"],
                # Combined context for model input
                "combined_text": f"Title: {title} | Parent: {parent_text} | Comment: {comment['comment']}"
            })
            
    return pd.DataFrame(rows)

if __name__ == "__main__":
    base_path = "Documents/Falak/Projects/LoFaCa/data"
    output_path = "Documents/Falak/Projects/LoFaCa/processed"
    os.makedirs(output_path, exist_ok=True)
    
    for split in ["train", "dev", "test"]:
        input_file = f"{base_path}/{split}.json"
        if os.path.exists(input_file):
            print(f"Processing {split}...")
            df = preprocess_cocolofa(input_file)
            df.to_csv(f"{output_path}/{split}_processed.csv", index=False)
            print(f"Saved {split}_processed.csv with {len(df)} rows.")

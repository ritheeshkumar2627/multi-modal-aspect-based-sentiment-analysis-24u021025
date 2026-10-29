import os
from collections import Counter
from configs.config import Config
from src.data_loader import parse_mabsa_txt

def main():
    for split, path in [("train", Config.TRAIN_FILE), ("dev", Config.DEV_FILE), ("test", Config.TEST_FILE)]:
        if not os.path.isfile(path):
            print(f"{split}: MISSING FILE: {path}")
            continue
        df = parse_mabsa_txt(path)
        print(f"\n{split}: {len(df):,} records")
        counts = Counter(df["raw_sentiment"].tolist())
        print("Original label counts (-1=negative, 0=neutral, 1=positive):", dict(counts))
        missing = sum(not os.path.isfile(os.path.join(Config.IMAGE_DIR, name)) for name in df["image_name"])
        print(f"Missing image files: {missing:,} / {len(df):,}")
    print("\nIf image files are reported missing, fix the image folder or filenames before training.")

if __name__ == "__main__":
    main()

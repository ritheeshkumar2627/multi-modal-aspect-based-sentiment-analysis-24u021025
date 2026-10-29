import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor, get_linear_schedule_with_warmup
from tqdm import tqdm

from configs.config import Config
from src.data_loader import parse_mabsa_txt, MABSADataset
from src.model import MABSAFusionModel
from src.utils import compute_metrics, make_classification_report

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, loader, device):
    model.eval()
    all_labels, all_predictions = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            pixels = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids, mask, pixels)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = compute_metrics(all_labels, all_predictions)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics, all_labels, all_predictions

def main():
    set_seed()
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df = parse_mabsa_txt(Config.TRAIN_FILE)
    dev_df = parse_mabsa_txt(Config.DEV_FILE)
    print(f"Training examples: {len(train_df):,}; validation examples: {len(dev_df):,}")
    if train_df.empty or dev_df.empty:
        raise ValueError("train.txt and dev.txt must both contain at least one example.")

    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)
    image_processor = AutoImageProcessor.from_pretrained(Config.VISION_MODEL_NAME)
    train_ds = MABSADataset(train_df, tokenizer, image_processor, Config.IMAGE_DIR)
    dev_ds = MABSADataset(dev_df, tokenizer, image_processor, Config.IMAGE_DIR)
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=(device.type == "cuda")
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=(device.type == "cuda")
    )

    model = MABSAFusionModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    total_steps = max(len(train_loader) * Config.EPOCHS, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_macro_f1 = -1.0
    history = []

    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            pixels = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids, mask, pixels)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        val_metrics, val_labels, val_preds = evaluate(model, dev_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(Config.HISTORY_PATH, index=False)
        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_accuracy={val_metrics['accuracy']:.4f}, "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "text_model_name": Config.TEXT_MODEL_NAME,
                "vision_model_name": Config.VISION_MODEL_NAME,
                "max_len": Config.MAX_LEN,
                "best_val_macro_f1": best_macro_f1,
                "label_map": Config.LABEL_MAP,
            }, Config.CHECKPOINT_PATH)
            print(f"Saved best checkpoint: {Config.CHECKPOINT_PATH}")

    print("\nBest validation macro-F1:", f"{best_macro_f1:.4f}")
    print("\nClassification report for the final epoch (validation set):")
    print(make_classification_report(val_labels, val_preds))
    print(f"History saved to: {Config.HISTORY_PATH}")
    print("Note: this is a baseline; benchmark scores are not guaranteed.")

if __name__ == "__main__":
    main()

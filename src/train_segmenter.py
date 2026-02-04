import json
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "segmentation" / "prepared"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"
OUT_DIR = ROOT / "models" / "segmenter"

MODEL_NAME = "aubmindlab/bert-base-arabertv2"  # قوي للعربي
MAX_LEN = 256
BATCH = 8
EPOCHS = 3
LR = 2e-5
SEED = 42


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    # تنظيف خفيف عشان الأداء يتحسن
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_lines(text: str):
    # تقسيم على أسطر + تنظيف
    raw = re.split(r"\n+", text)
    lines = [normalize(x) for x in raw if normalize(x)]
    return lines


def build_labels(text: str, spans):
    """
    نصنع label للسطر:
    1 إذا السطر يبدأ عند (أو قريب جدًا من) بداية span
    """
    # نحسب بدايات البنود
    starts = set([s for s, e in spans])

    # نحتاج نعرف offset لكل سطر داخل النص الأصلي
    # بنمشي على النص ونطلع الأسطر مع إزاحتها
    labels = []
    examples = []

    pos = 0
    for chunk in re.split(r"(\n+)", text):
        if chunk.startswith("\n"):
            pos += len(chunk)
            continue

        line = normalize(chunk)
        if not line:
            pos += len(chunk)
            continue

        # بداية هذا السطر في النص الأصلي
        start_offset = pos

        # نعتبره بداية بند إذا كان فيه span يبدأ قريب
        # (بعض البيانات يكون فيها مسافات/سطر جديد قبل البند)
        is_start = 0
        for s in starts:
            if abs(s - start_offset) <= 3:
                is_start = 1
                break

        examples.append(line)
        labels.append(is_start)

        pos += len(chunk)

    return examples, labels


class LinesDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.tokenizer = tokenizer
        self.texts = []
        self.labels = []

        for r in rows:
            text = r["text"]
            spans = r["spans"]
            lines, y = build_labels(text, spans)
            self.texts.extend(lines)
            self.labels.extend(y)

        enc = tokenizer(
            self.texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
        )
        self.encodings = enc

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def main():
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        print("❌ train/val not found. Run prepare_segmentation_data.py first.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(TRAIN_PATH)
    val_rows = load_jsonl(VAL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_ds = LinesDataset(train_rows, tokenizer)
    val_ds = LinesDataset(val_rows, tokenizer)

    args = TrainingArguments(
    output_dir=str(OUT_DIR),
    learning_rate=LR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=50,
    seed=SEED,
    fp16=False,
    report_to="none",
)


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    print(f"\n✅ Saved model to: {OUT_DIR}")


if __name__ == "__main__":
    main()

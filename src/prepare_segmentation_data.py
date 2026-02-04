import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "segmentation" / "segmented_contracts_dataset.json"
OUT_DIR = ROOT / "data" / "segmentation" / "prepared"

TRAIN_RATIO = 0.9
SEED = 42

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_item(item, idx: int):
    if "text" not in item or "spans" not in item:
        return False, f"Missing keys (text/spans) at idx={idx}"

    text = item["text"]
    spans = item["spans"]

    if not isinstance(text, str) or not isinstance(spans, list):
        return False, f"Bad types at idx={idx}"

    n = len(text)
    last_end = -1
    for s_idx, sp in enumerate(spans):
        if not (isinstance(sp, list) and len(sp) == 2):
            return False, f"Span not [start,end] at idx={idx}, span#{s_idx}"

        start, end = sp
        if not (isinstance(start, int) and isinstance(end, int)):
            return False, f"Span not int at idx={idx}, span#{s_idx}"

        if start < 0 or end > n or start >= end:
            return False, f"Span out of range at idx={idx}, span#{s_idx}: {sp} (text_len={n})"

        # Ensure non-overlap and sorted (common requirement)
        if start < last_end:
            return False, f"Overlapping/unsorted spans at idx={idx}, span#{s_idx}: {sp}"
        last_end = end

    # Optional sanity: extracted clauses shouldn't be empty after strip
    clauses = [text[a:b].strip() for a, b in spans]
    if any(c == "" for c in clauses):
        return False, f"Empty clause after strip at idx={idx}"

    return True, None

def spans_to_clauses(text: str, spans: list[list[int]]):
    return [text[a:b].strip() for a, b in spans]

def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    if not IN_PATH.exists():
        print(f"❌ File not found: {IN_PATH}")
        print("Make sure you placed the dataset here:")
        print("data/segmentation/segmented_contracts_dataset.json")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_json(IN_PATH)
    if not isinstance(data, list):
        print("❌ Dataset must be a JSON list of items.")
        return

    ok_items = []
    bad = 0

    for i, item in enumerate(data):
        valid, err = validate_item(item, i)
        if not valid:
            bad += 1
            if bad <= 10:
                print(f"[BAD] {err}")
            continue

        text = item["text"]
        spans = item["spans"]
        clauses = spans_to_clauses(text, spans)

        ok_items.append({
            "id": i,
            "text": text,
            "spans": spans,
            "clauses": clauses,
            "num_clauses": len(clauses)
        })

    print(f"✅ Valid items: {len(ok_items)}")
    print(f"⚠️ Bad items skipped: {bad}")

    # shuffle + split
    random.seed(SEED)
    random.shuffle(ok_items)

    cut = int(len(ok_items) * TRAIN_RATIO)
    train = ok_items[:cut]
    val = ok_items[cut:]

    train_path = OUT_DIR / "train.jsonl"
    val_path = OUT_DIR / "val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    # Also create a flat "clauses" dataset (each clause as one row) for later use
    flat = []
    for item in ok_items:
        for j, c in enumerate(item["clauses"]):
            flat.append({"doc_id": item["id"], "clause_id": j, "clause_text": c})

    flat_path = OUT_DIR / "clauses_flat.jsonl"
    write_jsonl(flat_path, flat)

    print("\n📦 Outputs:")
    print(f"- {train_path}")
    print(f"- {val_path}")
    print(f"- {flat_path}")
    print("\nNext step: we train a segmentation model using train/val.jsonl ✅")

if __name__ == "__main__":
    main()

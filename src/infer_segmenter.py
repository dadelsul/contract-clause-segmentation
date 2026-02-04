import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "segmenter"

MAX_LEN = 256

def normalize(text: str) -> str:
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_into_lines_with_offsets(raw_text: str):
    """
    يرجع قائمة:
    [(line_text, start_offset, end_offset), ...]
    اعتمادًا على تقسيم الأسطر في النص الأصلي
    """
    items = []
    pos = 0
    for chunk in re.split(r"(\n+)", raw_text):
        if chunk.startswith("\n"):
            pos += len(chunk)
            continue

        start = pos
        end = pos + len(chunk)
        line = normalize(chunk)

        if line:
            items.append((line, start, end))

        pos += len(chunk)
    return items

@torch.no_grad()
def predict_starts(raw_text: str, threshold: float = 0.5):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    lines = split_into_lines_with_offsets(raw_text)
    if not lines:
        return [], []

    texts = [t for (t, _, _) in lines]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")

    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[:, 1]  # class 1 = start
    preds = (probs >= threshold).long().tolist()

    return lines, list(zip(preds, probs.tolist()))

def build_clauses_from_starts(raw_text: str, lines, pred_info):
    """
    نبني البنود بتجميع الأسطر:
    كل ما جاء Start=1 نبدأ بند جديد.
    """
    clauses = []
    current = []

    for (line, _, _), (pred, prob) in zip(lines, pred_info):
        # إذا هذا بداية بند ومو أول سطر → اقفل البند السابق
        if pred == 1 and current:
            clauses.append("\n".join(current).strip())
            current = []

        current.append(line)

    if current:
        clauses.append("\n".join(current).strip())

    # تنظيف البنود الفاضية
    clauses = [c for c in clauses if c.strip()]
    return clauses

def demo_text():
    # مثال بسيط للعرض (تقدرين تغيرينه)
    return """أولاً: مدة العقد
تبدأ مدة العقد من تاريخ التوقيع ولمدة سنة قابلة للتجديد.

ثانياً: قيمة العقد
اتفق الطرفان على مبلغ وقدره (10000) ريال سعودي.

ثالثاً: التزامات الطرف الأول
يلتزم الطرف الأول بتقديم الخدمة وفق المواصفات المتفق عليها.

رابعاً: فسخ العقد
يجوز لأي طرف فسخ العقد في حال الإخلال بالشروط بعد إشعار كتابي.
"""

def main():
    raw_text = demo_text()

    lines, pred_info = predict_starts(raw_text, threshold=0.5)
    clauses = build_clauses_from_starts(raw_text, lines, pred_info)

    print("\n================= INPUT TEXT =================\n")
    print(raw_text)

    print("\n================= PREDICTED CLAUSES =================\n")
    for i, c in enumerate(clauses, 1):
        print(f"\n--- Clause {i} ---")
        print(c)

    # ✅ save to file (inside main)
    out_path = ROOT / "outputs" / "demo_output.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("INPUT TEXT\n")
        f.write(raw_text + "\n\n")
        f.write("PREDICTED CLAUSES\n\n")
        for i, c in enumerate(clauses, 1):
            f.write(f"--- Clause {i} ---\n")
            f.write(c + "\n\n")

    print(f"\n✅ Saved demo to: {out_path}")


if __name__ == "__main__":
    main()

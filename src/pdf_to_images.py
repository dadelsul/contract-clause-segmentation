from pathlib import Path
import re
import fitz  # pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "raw_pdfs"
OUT_DIR = ROOT / "data" / "images"

DPI = 200

def safe_stem(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\u0600-\u06FF\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "contract"

def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 200) -> int:
    doc = fitz.open(pdf_path)
    stem = safe_stem(pdf_path.stem)

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    saved = 0
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir / f"{stem}_p{str(i+1).zfill(3)}.png"
        pix.save(out_path.as_posix())
        saved += 1

    doc.close()
    return saved

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))

    if not pdfs:
        print("❌ No PDF files found.")
        return

    total = 0
    for idx, pdf in enumerate(pdfs, 1):
        try:
            pages = pdf_to_images(pdf, OUT_DIR, DPI)
            total += pages
            print(f"[{idx}/{len(pdfs)}] {pdf.name} → {pages} pages")
        except Exception as e:
            print(f"[ERROR] {pdf.name}: {e}")

    print(f"\n✅ Done. Total pages saved: {total}")

if __name__ == "__main__":
    main()

from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "data" / "images"
OUT_DIR = ROOT / "data" / "sample_images"

N = 30  # غيريها لو تبين 50 مثلا

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imgs = list(IMAGES_DIR.glob("*.png"))
    if not imgs:
        print("No PNG images found in data/images")
        return

    random.seed(42)
    picked = random.sample(imgs, k=min(N, len(imgs)))

    for p in picked:
        shutil.copy2(p, OUT_DIR / p.name)

    print(f"✅ Copied {len(picked)} images to: {OUT_DIR}")

if __name__ == "__main__":
    main()

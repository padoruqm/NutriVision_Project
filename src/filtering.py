import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.config import TARGET_SIZE, FOOD101_SUBSET, FOOD101_PROCESSED
def full_preprocess(img_bgr, angle=0, target_size=TARGET_SIZE):
    img = cv2.resize(img_bgr, target_size)
    if angle:
        h, w = target_size
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, target_size, borderMode=cv2.BORDER_REFLECT)
    img = cv2.bilateralFilter(img, d=7, sigmaColor=60, sigmaSpace=60)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return img.astype(np.float32) / 255.0

def show_preprocess_steps(img_bgr, angle=0, save_path=None):
    import matplotlib.pyplot as plt
    img = cv2.resize(img_bgr, TARGET_SIZE)
    if angle:
        center = (TARGET_SIZE[0] // 2, TARGET_SIZE[1] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, TARGET_SIZE, borderMode=cv2.BORDER_REFLECT)
    blurred = cv2.bilateralFilter(img, d=7, sigmaColor=60, sigmaSpace=60)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.5, (8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_Lab2BGR)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    steps = [
        (img, "Resized"),
        (blurred, "Bilateral"),
        (enhanced, "CLAHE"),
    ]
    for i, (im, title) in enumerate(steps):
        ax[i].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax[i].set_title(title)
        ax[i].axis("off")
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

def process_dataset(src_root=None, dst_root=None):
    src = Path(src_root) if src_root else FOOD101_SUBSET
    dst = Path(dst_root) if dst_root else FOOD101_PROCESSED
    imgs = list(src.rglob("*.jpg"))
    print(f"Found {len(imgs)} images")
    skipped = 0
    for p in tqdm(imgs):
        img = cv2.imread(str(p))
        if img is None:
            skipped += 1
            continue
        rel = p.relative_to(src)
        out_path = dst / rel.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        processed = full_preprocess(img)
        np.save(out_path, processed)

if __name__ == "__main__":
    process_dataset()
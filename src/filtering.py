import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
TARGET_SIZE = (224, 224)
def full_preprocess(img_bgr, angle=0, visualize=False):
    img = cv2.resize(img_bgr, TARGET_SIZE)
    if angle != 0:
        center = (TARGET_SIZE[0] // 2, TARGET_SIZE[1] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, TARGET_SIZE, borderMode=cv2.BORDER_REFLECT)
    img_bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_enhanced = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    img_norm = img_enhanced.astype(np.float32) / 255.0
    if visualize:
        _show_steps(img, img_bilateral, img_enhanced)
    return img_norm

def _show_steps(orig, after_filter, after_clahe):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, img, title in zip(ax,
        [orig, after_filter, after_clahe],
        ["Original (resized)", "After Bilateral", "After CLAHE"]):
        a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        a.set_title(title); a.axis("off")
    plt.tight_layout(); plt.show()

def process_dataset(src_root=r"C:\Users\ADMIN\Downloads\food101_subset",
                    dst_root=r"C:\Users\ADMIN\Downloads\food101_processed"):
    src = Path(src_root)
    dst = Path(dst_root)
    for img_path in tqdm(list(src.rglob("*.jpg"))):
        rel = img_path.relative_to(src)      
        out_path = dst / rel.with_suffix(".npy") 
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        processed = full_preprocess(img)
        np.save(out_path, processed)
    print(dst_root)
sample = cv2.imread(r"C:\Users\ADMIN\Downloads\food101_subset\train\pizza\5764.jpg")
full_preprocess(sample, angle = 30, visualize=True)
process_dataset()

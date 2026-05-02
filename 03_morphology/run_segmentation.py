import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import sys

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import (
    FOOD101_PROCESSED, 
    FOOD101_SEGMENTATION, 
    FOOD101_SUBSET,
    OUTPUTS_DIR
)
from segmentation import apply_grabcut_mask, overlay_mask
from morphology import FoodSegmenter
IMG_SAMPLE_DIR = OUTPUTS_DIR / "03_morphology" / "images"


def run_segmentation_pipeline():
    """Chạy GrabCut Hybrid segmentation + lưu crop + mask"""
    
    # Tạo thư mục chính và thư mục preview
    FOOD101_SEGMENTATION.mkdir(parents=True, exist_ok=True)
    IMG_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Đang chạy segmentation → Lưu chính vào: {FOOD101_SEGMENTATION}")
    print(f"   (5 ảnh mẫu preview sẽ được lưu thêm vào: {IMG_SAMPLE_DIR})")

    segmenter = FoodSegmenter()
    all_images = list(FOOD101_PROCESSED.rglob("*.jpg"))
    
    if not all_images:
        print("Không tìm thấy ảnh Processed!")
        return

    # Chọn 5 ảnh mẫu để preview
    sample_to_preview = random.sample(all_images, min(5, len(all_images)))

    for img_path in tqdm(all_images, desc="Đang segmentation (GrabCut Hybrid)"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        processed_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. Chạy GrabCut Hybrid
        binary_mask, bbox = apply_grabcut_mask(processed_rgb, filename=img_path.name)

        # 2. Tạo tight crop cho HOG
        crop_rgb = segmenter.extract_roi_for_hog(processed_rgb, None, bbox=bbox)

        # 3. Lưu file chính vào FOOD101_SEGMENTATION
        class_name = img_path.parent.name
        dest_dir = FOOD101_SEGMENTATION / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        base_name = img_path.stem

        cv2.imwrite(str(dest_dir / f"crop_{base_name}.jpg"), 
                    cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))

        if binary_mask is not None:
            cv2.imwrite(str(dest_dir / f"mask_{base_name}.png"), binary_mask)

        # 4. Lưu thêm 5 ảnh mẫu vào thư mục preview 
        if img_path in sample_to_preview:
            # Lưu crop và mask vào thư mục preview (tên giống hệt)
            cv2.imwrite(str(IMG_SAMPLE_DIR / f"crop_{base_name}.jpg"), 
                        cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
            
            if binary_mask is not None:
                cv2.imwrite(str(IMG_SAMPLE_DIR / f"mask_{base_name}.png"), binary_mask)

            # (Tùy chọn) Lưu overlay visualization
            vis = overlay_mask(processed_rgb, binary_mask, bbox)
            cv2.imwrite(str(IMG_SAMPLE_DIR / f"vis_{base_name}.jpg"), 
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"\nHOÀN THÀNH!")
    print(f"   - Tight crop (cho HOG):       {FOOD101_SEGMENTATION}/.../crop_*.jpg")
    print(f"   - Binary mask:                 mask_*.png")
    print(f"   - 5 ảnh mẫu preview (crop + mask + vis):")
    print(f"         → {IMG_SAMPLE_DIR}")
    print(f"   Tổng số lớp đã xử lý: {len(list(FOOD101_SEGMENTATION.iterdir()))}")


if __name__ == "__main__":
    run_segmentation_pipeline()
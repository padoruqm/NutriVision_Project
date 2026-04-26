import cv2
import sys
import os
import random
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import FOOD101_PROCESSED, OUTPUTS_DIR, DATA_DIR
FOOD101_CONTOUR = DATA_DIR / "food101_contour"
IMG_SAMPLE_DIR = OUTPUTS_DIR / "03_morphology" / "images"
from morphology import FoodSegmenter

def run_morphology_pipeline():
    FOOD101_CONTOUR.mkdir(parents=True, exist_ok=True)
    IMG_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Nguồn: {FOOD101_PROCESSED}")
    print(f"Đích: {FOOD101_CONTOUR}")

    segmenter = FoodSegmenter(target_size=(224, 224)) 
    # Lấy danh sách tất cả ảnh đã tiền xử lý
    all_images = list(FOOD101_PROCESSED.rglob("*.jpg"))
    if not all_images:
        print("Không tìm thấy dữ liệu tại 02_preprocessing. Hãy chạy step 02 trước!")
        return

    # Biến để lưu mẫu in ra (5 ảnh)
    sample_to_save = random.sample(all_images, min(5, len(all_images)))
    
    # 2. Chạy vòng lặp xử lý toàn bộ dataset
    for img_path in tqdm(all_images, desc="Đang trích xuất Contour"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue

        # Áp dụng strategy 'gray_canny'
        _, binary, contour = segmenter.get_mask(img_bgr, strategy='gray_canny')

        # Vẽ contour lên ảnh để lưu kết quả trực quan
        res_img = img_bgr.copy()
        if contour is not None:
            cv2.drawContours(res_img, [contour], -1, (0, 255, 0), 2)

        # Lưu vào data/food101_contour 
        class_name = img_path.parent.name
        dest_dir = FOOD101_CONTOUR / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(dest_dir / img_path.name), res_img)

        # 3. Nếu là ảnh mẫu, vẽ so sánh và lưu vào outputs
        if img_path in sample_to_save:
            save_comparison(img_bgr, res_img, img_path.name)

    print(f"\nHoàn thành! Đã lưu 5 ảnh ví dụ tại: {IMG_SAMPLE_DIR}")

def save_comparison(original_pre, contoured_img, file_name):
    """Lưu ảnh so sánh: Preprocessed vs Contoured"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Chuyển BGR sang RGB để matplotlib hiển thị đúng
    axes[0].imshow(cv2.cvtColor(original_pre, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Preprocessed Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(contoured_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("2. Gray Canny Contour", fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(IMG_SAMPLE_DIR / f"cmp_{file_name}")
    plt.close()

if __name__ == "__main__":
    run_morphology_pipeline()
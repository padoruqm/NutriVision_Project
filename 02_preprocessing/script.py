import cv2
import sys
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding='utf-8')
from src.config import FOOD101_SUBSET, TARGET_SIZE, OUTPUTS_DIR, FOOD101_PROCESSED
import importlib
SAVE_DIR = FOOD101_PROCESSED
IMG_SAMPLE_DIR = OUTPUTS_DIR / "02_preprocessing" / "images"
preprocessing_module = importlib.import_module("02_preprocessing.preprocessing")
ImageEnhancer = preprocessing_module.ImageEnhancer

def run_preprocessing_pipeline():
    # Tạo các thư mục đích
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    IMG_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Đích đến: {SAVE_DIR}")
    
    enhancer = ImageEnhancer(target_size=TARGET_SIZE)
    all_images = list(FOOD101_SUBSET.rglob("*.jpg"))
    
    if not all_images:
        print("Không tìm thấy ảnh!")
        return

    # Chọn ngẫu nhiên 5 ảnh để làm mẫu hiển thị/lưu riêng
    sample_paths = random.sample(all_images, min(5, len(all_images)))

    # Chạy vòng lặp xử lý toàn bộ tập dữ liệu
    for img_path in tqdm(all_images, desc="Đang xử lý toàn bộ dataset"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
            
        # Thực hiện tiền xử lý
        enhanced_img = enhancer.enhance(img_bgr)
        
        # Lưu ảnh vào SAVE_DIR (giữ nguyên cấu trúc thư mục con)
        rel_path = img_path.relative_to(FOOD101_SUBSET)
        dest_path = SAVE_DIR / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest_path), cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))
    
    print(f"so sánh tổng hợp tại: {IMG_SAMPLE_DIR}")
    for i, img_path in enumerate(sample_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue

        # === Tính theo đúng pipeline enhancer ===
        enhanced_rgb = enhancer.enhance(img_bgr)                    # Full pipeline
        blurred = cv2.bilateralFilter(img_bgr, 5, 50, 50)
        
        # Histogram (trước & sau CLAHE)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l = lab[:,:,0]
        hist_before = cv2.calcHist([l], [0], None, [256], [0, 256])
        l_enhanced = enhancer.clahe.apply(l)
        hist_after = cv2.calcHist([l_enhanced], [0], None, [256], [0, 256])

        # Chuyển về RGB để matplotlib
        original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        blurred_rgb   = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        final_rgb     = enhanced_rgb

        fig = plt.figure(figsize=(16, 9), layout="constrained")
        fig.canvas.manager.set_window_title(f"Test {i+1}: {img_path.name}")

        ax1 = plt.subplot2grid((2, 4), (0, 0))
        ax1.imshow(original_rgb)
        ax1.set_title("1. Original", fontweight='bold')
        ax1.axis("off")

        ax2 = plt.subplot2grid((2, 4), (0, 1))
        ax2.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cmap='gray')
        ax2.set_title("2. Grayscale (Original)", fontweight='bold')
        ax2.axis("off")

        ax3 = plt.subplot2grid((2, 4), (0, 2))
        ax3.imshow(blurred_rgb)
        ax3.set_title("3. Bilateral Blur", fontweight='bold')
        ax3.axis("off")

        ax4 = plt.subplot2grid((2, 4), (0, 3))
        ax4.imshow(final_rgb)
        ax4.set_title("4. Final (CLAHE + Resize)", fontweight='bold')
        ax4.axis("off")

        # Histogram (giữ nguyên code của bạn)
        ax5 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
        ax5.plot(hist_before, color='gray', linewidth=2)
        ax5.fill_between(range(256), hist_before.flatten(), color='gray', alpha=0.3)
        ax5.set_title("Histogram Kênh L (Trước CLAHE)")
        ax5.set_xlim([0, 256])
        ax5.grid(True, linestyle='--', alpha=0.5)

        ax6 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
        ax6.plot(hist_after, color='blue', linewidth=2)
        ax6.fill_between(range(256), hist_after.flatten(), color='blue', alpha=0.3)
        ax6.set_title("Histogram Kênh L (Sau CLAHE)")
        ax6.set_xlim([0, 256])
        ax6.grid(True, linestyle='--', alpha=0.5)

        analysis_out_path = IMG_SAMPLE_DIR / f"analysis_{img_path.stem}.png"
        plt.savefig(analysis_out_path, dpi=150)
        plt.close(fig)

    print(f"\n HOÀN THÀNH!")
    print(f"- Toàn bộ ảnh lưu tại: {SAVE_DIR}")
    print(f"- 5 ảnh mẫu lưu tại: {IMG_SAMPLE_DIR}")

if __name__ == "__main__":
    run_preprocessing_pipeline()
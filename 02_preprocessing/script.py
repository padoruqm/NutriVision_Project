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
        cv2.imwrite(str(dest_path), enhanced_img)
    
    print(f"so sánh tổng hợp tại: {IMG_SAMPLE_DIR}")
    for i, img_path in enumerate(sample_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue

        # --- Tách bóc từng bước để vẽ ảnh và đồ thị ---
        # 1. Original (Đã resize padding)
        img_resized = enhancer.resize_keep_ratio(img_bgr)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # 2. Grayscale 
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # 3. Blur (Bilateral Filter)
        blurred = cv2.bilateralFilter(img_resized, d=5, sigmaColor=50, sigmaSpace=50)
        blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

        # 4. Tách kênh L để phân tích Histogram và chạy CLAHE (Final)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Tính Histogram trước khi CLAHE
        hist_l_before = cv2.calcHist([l], [0], None, [256], [0, 256])
        
        # Chạy CLAHE
        l_enhanced = enhancer.clahe.apply(l)
        
        # Tính Histogram sau khi CLAHE
        hist_l_after = cv2.calcHist([l_enhanced], [0], None, [256], [0, 256])
        
        # Ghép lại thành ảnh Final
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        final_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
        
        fig = plt.figure(figsize=(16, 9), layout="constrained")
        fig.canvas.manager.set_window_title(f"Test {i+1}: {img_path.name} - Phân tích CLAHE")

        # --- HÀNG 1: 4 ẢNH (Chia làm 4 cột) ---
        ax1 = plt.subplot2grid((2, 4), (0, 0))
        ax1.imshow(img_rgb)
        ax1.set_title("1. Original (Resized)", fontweight='bold')
        ax1.axis("off")

        ax2 = plt.subplot2grid((2, 4), (0, 1))
        ax2.imshow(gray, cmap='gray')
        ax2.set_title("2. Grayscale", fontweight='bold')
        ax2.axis("off")

        ax3 = plt.subplot2grid((2, 4), (0, 2))
        ax3.imshow(blurred_rgb)
        ax3.set_title("3. Blur (Bilateral)", fontweight='bold')
        ax3.axis("off")

        ax4 = plt.subplot2grid((2, 4), (0, 3))
        ax4.imshow(final_rgb)
        ax4.set_title("4. Final (CLAHE)", fontweight='bold')
        ax4.axis("off")

        # --- HÀNG 2: 2 ĐỒ THỊ HISTOGRAM (Mỗi đồ thị chiếm 2 cột ngang) ---
        ax5 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
        ax5.plot(hist_l_before, color='gray', linewidth=2)
        ax5.fill_between(range(256), hist_l_before.flatten(), color='gray', alpha=0.3)
        ax5.set_title("Histogram Kênh L (Trạng thái gốc - Mờ nhạt)", fontweight='bold')
        ax5.set_xlim([0, 256])
        ax5.set_xlabel("Độ sáng (0: Đen -> 255: Trắng)")
        ax5.set_ylabel("Số lượng Pixel")
        ax5.grid(True, linestyle='--', alpha=0.5)

        ax6 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
        ax6.plot(hist_l_after, color='blue', linewidth=2)
        ax6.fill_between(range(256), hist_l_after.flatten(), color='blue', alpha=0.3)
        ax6.set_title("Histogram Kênh L (Sau CLAHE - Trải rộng, Tăng tương phản)", fontweight='bold')
        ax6.set_xlim([0, 256])
        ax6.set_xlabel("Độ sáng (0: Đen -> 255: Trắng)")
        ax6.grid(True, linestyle='--', alpha=0.5)
        analysis_out_path = IMG_SAMPLE_DIR / f"analysis_{img_path.stem}.png"
        plt.savefig(analysis_out_path, dpi=150)
        plt.close(fig) 

    print(f"\n HOÀN THÀNH!")
    print(f"- Toàn bộ ảnh lưu tại: {SAVE_DIR}")
    print(f"- 5 ảnh mẫu lưu tại: {IMG_SAMPLE_DIR}")

if __name__ == "__main__":
    run_preprocessing_pipeline()
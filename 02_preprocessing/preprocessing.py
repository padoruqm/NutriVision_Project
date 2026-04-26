import zipfile
import cv2
import gdown
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import os

class ImageEnhancer:
    def __init__(self, target_size=(224, 224), clip_limit=1.5):
        self.target_size = target_size
        # Khởi tạo CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    def resize_keep_ratio(self, img):
        """
        Resize ảnh màu bgr về size x size nhưng vẫn giữ tỉ lệ gốc, đệm padding màu đen nếu cần."""
        target_w, target_h = self.target_size
        h, w = img.shape[:2]
        
        # Tính tỷ lệ theo cạnh dài nhất để ảnh không bị tràn canvas
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize dùng INTER_AREA để giảm thiểu mất mát chi tiết khi giảm kích thước
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Tạo canvas nền đen
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Tính toán để căn giữa ảnh trên canvas
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Dán ảnh vào giữa canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        
        return canvas
    def enhance(self, img_bgr):
        """
        Pipeline: Resize -> Denoise -> CLAHE 
        """
        if img_bgr is None: return None

        # 1. Resize 
        img_resized = self.resize_keep_ratio(img_bgr)

        # 2. Denoise bằng Bilateral Filter
        blurred = cv2.bilateralFilter(img_resized, d=5, sigmaColor=50, sigmaSpace=50)
        # 3. Contrast Enhancement (CLAHE trên kênh L của Lab để bảo vệ màu sắc)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return img_enhanced
if __name__ == "__main__":
    ZIP_ID = "1Y4LQIhG1dKseOuPiy4w9bX2y4JjRVZEt"
    DATA_DIR = "./train_dataset"
    ZIP_PATH = "./train_data.zip"

    if not os.path.exists(DATA_DIR):
        gdown.download(f'https://drive.google.com/uc?id={ZIP_ID}', ZIP_PATH, quiet=False)
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(DATA_DIR)
        os.remove(ZIP_PATH)

    if not os.path.exists(DATA_DIR):
        print(f"Không tìm thấy thư mục {DATA_DIR}. Hãy chạy file chính để tải data trước.")
    else:
        all_images = list(Path(DATA_DIR).rglob("*.jpg"))
        
        # Lấy ngẫu nhiên 5 ảnh để test
        samples = random.sample(all_images, min(5, len(all_images)))
        enhancer = ImageEnhancer()

        for i, img_p in enumerate(samples):
            img_bgr = cv2.imread(str(img_p))
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
            fig.canvas.manager.set_window_title(f"Test {i+1}: {img_p.name} - Phân tích CLAHE")

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

            plt.show() # Tắt cửa sổ để sang ảnh tiếp theo
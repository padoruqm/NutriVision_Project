import cv2
import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
sys.stdout.reconfigure(encoding='utf-8')
from src.config import FOOD101_PROCESSED
class FoodSegmenter:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def auto_canny(self, image, sigma=0.33):
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(image, lower, upper)
    def get_mask(self, clean_img, strategy='lab_canny'):
        """Thực hiện 4 chiến lược khác nhau, nhìn kết quả thủ công t chọn gray_canny vì nó có vẻ ổn định nhất trên nhiều ảnh, còn 3 chiến lược kia đôi khi bị thiếu hoặc thừa biên"""
        binary = None
        if strategy == 'gray_canny':
            gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
            binary = self.auto_canny(gray)
            
        elif strategy == 'otsu_sv':
            hsv = cv2.cvtColor(clean_img, cv2.COLOR_BGR2HSV)
            _, s_mask = cv2.threshold(hsv[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, v_mask = cv2.threshold(hsv[:,:,2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.bitwise_and(s_mask, v_mask)
            
        elif strategy == 'lab_canny':
            lab = cv2.cvtColor(clean_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            binary = np.maximum(self.auto_canny(l), np.maximum(self.auto_canny(a), self.auto_canny(b)))
            
        elif strategy == 'gradient_canny':
            # CHIẾN LƯỢC 4: Canny trên Color Gradient (Vector Gradient)
            # Tính đạo hàm Sobel trên cả 3 kênh màu
            img_f = clean_img.astype(np.float32)
            gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
            # Tính biên độ gradient tổng hợp: mag = sqrt(gx^2 + gy^2)
            mag = cv2.sqrt(gx**2 + gy**2)
            # Lấy giá trị lớn nhất trong 3 kênh màu tại mỗi pixel
            color_grad = np.max(mag, axis=2).astype(np.uint8)
            binary = self.auto_canny(color_grad)

        # Morphology để đóng kín các đường biên
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Tìm contour lớn nhất
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None
        img_area = binary.shape[0] * binary.shape[1]
        
        if contours:
            # Sắp xếp contour từ lớn đến bé
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Bỏ qua những contour quá to (chiếm > 90% ảnh, thường là viền khung ảnh)
                # Hoặc quá nhỏ (chiếm < 1% ảnh)
                if 0.01 * img_area < area < 0.90 * img_area:
                    best_cnt = cnt
                    break
            if best_cnt is None:
                best_cnt = contours[0]
        return clean_img, binary, best_cnt

def show_images(images, titles=None, cols=3, figsize=(10, 20), cmap='gray'):
    n = len(images)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, 
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.3})
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes] # Nếu chỉ có 1 ảnh

    for i, img in enumerate(images):
        ax = axes[i]
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap)
        elif img.ndim == 3 and img.shape[2] == 1:
            ax.imshow(img[:, :, 0], cmap=cmap)
        else:
            ax.imshow(img)

        ax.axis('off')
        if titles is not None and i < len(titles):
            ax.set_title(titles[i], fontsize=12, fontweight='bold', pad=8)
    for j in range(len(images), len(axes)):
        axes[j].axis('off')

    plt.show()
if __name__ == "__main__":
    print(f"Đang chạy kiểm tra 50 ảnh với 4 phương pháp Contour...")
    print(f"Nguồn dữ liệu (Đã tiền xử lý): {FOOD101_PROCESSED}")
    
    segmenter = FoodSegmenter()
    all_images = list(FOOD101_PROCESSED.rglob("*.jpg"))
    
    if not all_images:
        print("Không tìm thấy ảnh! Hãy chạy Step 02: Preprocessing trước.")
        sys.exit()
        
    samples = random.sample(all_images, min(50, len(all_images)))
    strategies = ['gray_canny', 'otsu_sv', 'lab_canny', 'gradient_canny']
    
    for i, img_p in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] Hiển thị ảnh: {img_p.name}")
        # Đọc ảnh ĐÃ tiền xử lý
        clean_img = cv2.imread(str(img_p))
        if clean_img is None: continue
        
        clean_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        display_images = [clean_rgb, clean_rgb]
        display_titles = ["1. Input (Processed)", "2. Ready for Contour"]
        
        for st in strategies:
            _, binary, cnt = segmenter.get_mask(clean_img, strategy=st)
            
            # Vẽ viền xanh lá (0, 255, 0) lên ảnh
            vis = clean_img.copy()
            if cnt is not None:
                cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
                
            display_images.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            display_titles.append(f"Method: {st}")

        show_images(display_images, titles=display_titles, cols=6, figsize=(24, 4))
        print("Hãy đóng (tắt) cửa sổ ảnh để xem ảnh tiếp theo...")
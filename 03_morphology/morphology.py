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
    """
    Lớp chính thực hiện segmentation truyền thống dựa trên edge detection
    và region-based methods cho tập Food-101.
    """

    def __init__(self, target_size=(224, 224), default_strategy='combine_lab_otsu'):
        self.target_size = target_size
        self.default_strategy = default_strategy

    @staticmethod
    def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
        """Canny với ngưỡng tự động dựa trên median intensity."""
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(image, lower, upper)

    @staticmethod
    def _apply_morphology_close(binary: np.ndarray) -> np.ndarray:
        """Morphology closing để nối biên và loại nhiễu nhỏ."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def get_mask(self, clean_img_rgb: np.ndarray, strategy: str | None = None, filename: str = None):
        """
        Trả về (original_rgb, binary_mask, best_contour)
        strategy: 'gray_canny', 'otsu_sv', 'lab_canny', 'gradient_canny', 'combine_lab_otsu'
        """
        if strategy is None:
            strategy = self.default_strategy

        binary = None

        if strategy == 'gray_canny':
            gray = cv2.cvtColor(clean_img_rgb, cv2.COLOR_RGB2GRAY)
            binary = self.auto_canny(gray)

        elif strategy == 'otsu_sv':
            hsv = cv2.cvtColor(clean_img_rgb, cv2.COLOR_RGB2HSV)
            _, s_mask = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, v_mask = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.bitwise_and(s_mask, v_mask)

        elif strategy == 'lab_canny':
            lab = cv2.cvtColor(clean_img_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            binary = np.maximum(self.auto_canny(l), np.maximum(self.auto_canny(a), self.auto_canny(b)))

        elif strategy == 'gradient_canny':
            img_f = clean_img_rgb.astype(np.float32)
            gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.sqrt(gx**2 + gy**2)
            color_grad = np.max(mag, axis=2).astype(np.uint8)
            binary = self.auto_canny(color_grad)

        elif strategy == 'combine_lab_otsu':
            # Lab Canny
            lab = cv2.cvtColor(clean_img_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            binary_lab = np.maximum(self.auto_canny(l),
                                    np.maximum(self.auto_canny(a), self.auto_canny(b)))

            # Otsu trên HSV
            hsv = cv2.cvtColor(clean_img_rgb, cv2.COLOR_RGB2HSV)
            _, s_mask = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, v_mask = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_otsu = cv2.bitwise_and(s_mask, v_mask)

            # Close riêng từng mask trước khi union
            binary_lab = self._apply_morphology_close(binary_lab)
            binary_otsu = self._apply_morphology_close(binary_otsu)

            binary = cv2.bitwise_or(binary_lab, binary_otsu)

        else:
            raise ValueError(f"Strategy không hợp lệ: {strategy}")

        # Áp dụng morphology closing (trừ combine_lab_otsu đã close riêng)
        if strategy != 'combine_lab_otsu':
            binary = self._apply_morphology_close(binary)

        # Tìm contour lớn nhất (loại bỏ viền khung)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None
        img_area = binary.shape[0] * binary.shape[1]

        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Tăng ngưỡng upper lên 95% để chấp nhận contour lớn hơn
                if 0.01 * img_area < area < 0.95 * img_area:
                    best_cnt = cnt
                    break
            
            # Nếu vẫn không có contour thỏa mãn, lấy contour lớn nhất nhưng cảnh báo
            if best_cnt is None:
                best_cnt = contours[0]
                area_ratio = cv2.contourArea(best_cnt) / img_area * 100
                warning_msg = f"[Warning] Contour lớn nhất chiếm {area_ratio:.1f}%"
                if filename:
                    warning_msg += f" → {filename}"
                print(warning_msg + " → có thể crop không tight")
        return clean_img_rgb, binary, best_cnt
    def extract_roi_for_hog(self, clean_img_rgb: np.ndarray, best_cnt, bbox=None):
        """
        Tight crop từ contour (ưu tiên fill solid mask).
        """
        if best_cnt is None and bbox is None:
            return cv2.resize(clean_img_rgb, self.target_size)

        # Ưu tiên bbox từ clustering (nếu có)
        if bbox is not None:
            x, y, w, h = bbox
            if w > 0 and h > 0:
                crop = clean_img_rgb[y:y + h, x:x + w]
                return self._resize_keep_aspect(crop)

        if best_cnt is None:
            return cv2.resize(clean_img_rgb, self.target_size)

        # TẠO MASK SOLID TỪ CONTOUR
        h, w = clean_img_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [best_cnt], -1, 255, cv2.FILLED)   # Fill hoàn toàn bên trong

        # Erosion nhẹ (chỉ 1 iteration, kernel nhỏ) để tránh dính background
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)

        # Nếu erosion làm mất hết → fallback mask gốc
        if np.count_nonzero(eroded) < 100:   # ngưỡng nhỏ
            eroded = mask

        x, y, ww, hh = cv2.boundingRect(eroded)
        if ww == 0 or hh == 0:
            return cv2.resize(clean_img_rgb, self.target_size)

        crop = clean_img_rgb[y:y + hh, x:x + ww]
        return self._resize_keep_aspect(crop, padding_mode='mean_color')   

    def _resize_keep_aspect(self, img: np.ndarray, padding_mode: str = 'mean_color') -> np.ndarray:
        """
        Resize giữ tỷ lệ + padding.
        padding_mode: 'reflect' (cũ), 'black', 'mean_color' (khuyến nghị cho HOG)
        """
        h, w = img.shape[:2]
        target_w, target_h = self.target_size

        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        if padding_mode == 'black':
            color = (0, 0, 0)
        elif padding_mode == 'mean_color':
            color = cv2.mean(resized)[:3]          # BGR
            color = tuple(map(int, color))
        else:  # reflect
            return cv2.copyMakeBorder(resized, top, bottom, left, right,
                                      cv2.BORDER_REFLECT_101)

        return cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=color)
def show_images(images, titles=None, cols=3, figsize=(15, 10), cmap='gray'):
    """Hiển thị nhiều ảnh theo lưới."""
    n = len(images)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize,
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.3})
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, img in enumerate(images):
        ax = axes[i]
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=11, fontweight='bold', pad=8)

    for j in range(len(images), len(axes)):
        axes[j].axis('off')
    plt.show()

if __name__ == "__main__":
    print(f"Đang hiển thị mask overlay cho 5 strategies...")
    print(f"Nguồn: {FOOD101_PROCESSED}")

    segmenter = FoodSegmenter()
    all_images = list(FOOD101_PROCESSED.rglob("*.jpg"))
    if not all_images:
        print("Không tìm thấy ảnh! Hãy chạy Preprocessing trước.")
        sys.exit()

    samples = random.sample(all_images, min(50, len(all_images)))
    strategies = ['gray_canny', 'otsu_sv', 'lab_canny', 'gradient_canny', 'combine_lab_otsu']

    for i, img_p in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {img_p.name}")
        clean_img_bgr = cv2.imread(str(img_p))
        if clean_img_bgr is None:
            continue
        clean_img_rgb = cv2.cvtColor(clean_img_bgr, cv2.COLOR_BGR2RGB)
        h, w = clean_img_rgb.shape[:2]

        display_images = [clean_img_rgb]
        display_titles = ["1. Input (Processed)"]

        for st in strategies:
            _, _, cnt = segmenter.get_mask(clean_img_rgb, strategy=st)
            
            # Tạo solid mask từ contour (nếu có)
            solid_mask = np.zeros((h, w), dtype=np.uint8)
            if cnt is not None:
                cv2.drawContours(solid_mask, [cnt], -1, 255, cv2.FILLED)
            
            # Overlay: tô màu xanh (0,255,0) lên vùng mask với alpha=0.5
            overlay = clean_img_rgb.copy()
            color = np.array([0, 255, 0], dtype=np.uint8)
            mask_3ch = np.stack([solid_mask]*3, axis=2)
            alpha = 0.5
            # Chỉ vùng mask mới bị pha màu
            overlay = np.where(mask_3ch > 0, 
                               (alpha * color + (1 - alpha) * overlay).astype(np.uint8),
                               overlay)
            # Vẽ contour viền xanh đậm
            if cnt is not None:
                cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
            
            display_images.append(overlay)
            display_titles.append(f"{st}")

        show_images(display_images, titles=display_titles, cols=3, figsize=(18, 12))
        print("→ Đóng cửa sổ để xem ảnh tiếp theo...\n")
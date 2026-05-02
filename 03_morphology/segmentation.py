import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
sys.stdout.reconfigure(encoding='utf-8')

from src.config import FOOD101_PROCESSED, FOOD101_SUBSET
from morphology import FoodSegmenter  

def get_food_label_heuristic(labels):
    """Giả định Center-Crop: Lấy vùng lõi 40% để tìm nhãn phổ biến nhất"""
    try:
        h, w = labels.shape
        center_roi = labels[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
        valid_labels = center_roi[center_roi != -1]
        if len(valid_labels) == 0:
            return -1
        counts = np.bincount(valid_labels.flatten())
        return np.argmax(counts)
    except Exception as e:
        print(f"[Cảnh báo] Lỗi trong get_food_label_heuristic: {e}")
        return -1


def clean_mask_and_get_bbox(binary_mask):
    """Morphology làm sạch mask, tìm Contour lớn nhất và trả về BBox"""
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask_opened, None
            
        best_cnt = max(contours, key=cv2.contourArea)
        img_area = binary_mask.shape[0] * binary_mask.shape[1]
        
        if cv2.contourArea(best_cnt) < 0.01 * img_area:
            return mask_opened, None

        x, y, w, h = cv2.boundingRect(best_cnt)
        return mask_opened, (x, y, w, h)
    except Exception as e:
        print(f"[Cảnh báo] Lỗi trong clean_mask_and_get_bbox: {e}")
        return binary_mask, None


def overlay_mask(img_rgb, mask, bbox, color=(0, 255, 0), alpha=0.5):
    """Tạo overlay với mask được tô màu xanh trong suốt và vẽ bbox"""
    overlay = img_rgb.copy()
    mask_3ch = np.stack([mask] * 3, axis=2)
    colored_mask = np.zeros_like(img_rgb, dtype=np.uint8)
    colored_mask[:] = color
    overlay = np.where(mask_3ch > 0,
                       (alpha * colored_mask + (1 - alpha) * overlay).astype(np.uint8),
                       overlay)
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)
    return overlay

def apply_watershed_mask(img_rgb):
    """
    Watershed Segmentation Hybrid (S-channel + b-channel)
    """
    try:
        h, w = img_rgb.shape[:2]
        if img_rgb.size == 0:
            raise ValueError("Ảnh đầu vào rỗng")

        # 1. Hybrid color mask
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        s_channel = hsv[:, :, 1]
        b_channel = lab[:, :, 2]

        _, mask_s = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask_b = cv2.threshold(b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        combined_mask = cv2.bitwise_or(mask_s, mask_b)

        # 2. Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        opening = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # 3. Sure background & foreground
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        if dist_transform.max() == 0:
            return np.zeros((h, w), dtype=np.uint8), None

        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)

        # 4. Markers + Watershed
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  
        markers = cv2.watershed(img_bgr, markers)

        # 5. Final mask
        final_mask = np.zeros((h, w), dtype=np.uint8)
        final_mask[markers > 1] = 255

        # Cleanup cuối
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, 
                                      np.ones((5,5), np.uint8), iterations=2)

        return clean_mask_and_get_bbox(final_mask)

    except Exception as e:
        print(f"[-] Lỗi Watershed: {e}")
        return np.zeros(img_rgb.shape[:2], dtype=np.uint8), None

def apply_grabcut_mask(img_rgb, filename: str = None):
    """
    GrabCut hybrid: Sử dụng contour từ combine_lab_otsu làm initial mask
    """
    h, w = img_rgb.shape[:2]
    try:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # === Bước 1: Lấy contour chất lượng cao từ morphology ===
        segmenter = FoodSegmenter()
        _, _, best_cnt = segmenter.get_mask(img_rgb, strategy='combine_lab_otsu', filename=filename)
        
        # Khởi tạo mask GrabCut
        mask_gc = np.zeros((h, w), dtype=np.uint8)
        
        if best_cnt is not None:
            # Vùng bên trong contour = GC_FGD (foreground chắc chắn)
            cv2.drawContours(mask_gc, [best_cnt], -1, cv2.GC_FGD, thickness=cv2.FILLED)
            # Vùng ngoài contour một chút = GC_BGD (background chắc chắn)
            cv2.drawContours(mask_gc, [best_cnt], -1, cv2.GC_BGD, thickness=10)
        
        # Model cho GrabCut
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Chạy GrabCut với initial mask
        cv2.grabCut(img_bgr, mask_gc, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
        
        # Tạo binary mask
        binary_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD),
                               255, 0).astype(np.uint8)
        
        # Cleanup
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return clean_mask_and_get_bbox(binary_mask)
    except Exception as e:
        print(f"[-] Lỗi GrabCut Hybrid: {e}")
        return np.zeros((h, w), dtype=np.uint8), None


def run_clustering_pipeline():
    processed_images = list(FOOD101_PROCESSED.rglob("*.jpg"))
    if not processed_images:
        print("Không tìm thấy ảnh Processed!")
        return
        
    samples = random.sample(processed_images, min(80, len(processed_images)))  # giảm nhẹ để nhanh
    
    for i, img_path in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] Đang xử lý: {img_path.name}")
        try:
            processed_img = cv2.imread(str(img_path))
            if processed_img is None:
                raise ValueError("cv2.imread trả về None")
            processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            
            # Đọc ảnh gốc để so sánh
            raw_path = FOOD101_SUBSET / img_path.parent.name / img_path.name
            raw_rgb = cv2.cvtColor(cv2.imread(str(raw_path)), cv2.COLOR_BGR2RGB) if raw_path.exists() else processed_rgb
            
            # Chạy 2 phương pháp chính
            ws_mask, ws_bbox = apply_watershed_mask(processed_rgb)
            gc_mask, gc_bbox = apply_grabcut_mask(processed_rgb)
            
            # Overlay
            ws_overlay = overlay_mask(processed_rgb, ws_mask, ws_bbox)
            gc_overlay = overlay_mask(processed_rgb, gc_mask, gc_bbox)
            
            # Hiển thị 4 cột
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 4, 1); plt.imshow(raw_rgb); plt.title("1. Original"); plt.axis('off')
            plt.subplot(1, 4, 2); plt.imshow(processed_rgb); plt.title("2. Preprocessed"); plt.axis('off')
            plt.subplot(1, 4, 3); plt.imshow(ws_overlay); plt.title("3. Watershed"); plt.axis('off')
            plt.subplot(1, 4, 4); plt.imshow(gc_overlay); plt.title("4. GrabCut (Hybrid)"); plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"[-] Bỏ qua file {img_path.name} do lỗi: {e}")
            continue


if __name__ == "__main__":
    run_clustering_pipeline()
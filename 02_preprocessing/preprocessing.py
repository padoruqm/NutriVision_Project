import cv2
import numpy as np

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
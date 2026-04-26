import cv2
import numpy as np

class ImageEnhancer:
    def __init__(self, target_size=(224, 224), clip_limit=1.5):
        self.target_size = target_size
        # Khởi tạo CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    import cv2

    def resize_reflect_padding(self, img):
        target_w, target_h = self.target_size
        h, w = img.shape[:2]

        # Nếu đã đúng size thì bỏ qua
        if (w, h) == (target_w, target_h):
            return img

        # Scale để fit vào khung (giữ tỉ lệ)
        scale = min(target_w / w, target_h / h)
        new_w = round(w * scale)
        new_h = round(h * scale)

        # Resize
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # Reflect padding
        padded_img = cv2.copyMakeBorder(
            img_resized,
            top, bottom, left, right,
            cv2.BORDER_REFLECT_101
        )
        return padded_img
    def enhance(self, img_bgr):
        """
        Pipeline: Resize -> Denoise -> CLAHE 
        """
        if img_bgr is None: return None

        # 1. Resize 
        img_resized = self.resize_reflect_padding(img_bgr)

        # 2. Denoise bằng Bilateral Filter
        blurred = cv2.bilateralFilter(img_resized, d=5, sigmaColor=50, sigmaSpace=50)
        # 3. Contrast Enhancement (CLAHE trên kênh L của Lab để bảo vệ màu sắc)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return img_enhanced
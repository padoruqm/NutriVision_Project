
import cv2
import numpy as np
import pickle
from skimage.feature import hog

class ImageEnhancer:
    """
    Tích hợp từ 02_preprocessing/preprocessing.py – điều chỉnh thứ tự cho HOG.

    Pipeline enhance() (trên độ phân giải GỐC):
        Bilateral Filter  →  CLAHE trên kênh L (LAB)

    Resize được tách ra khỏi enhance() và thực hiện CUỐI CÙNG ở preprocess_image()
    sau khi đã grayscale, lý do:
        • Bilateral & CLAHE hoạt động tốt hơn ở độ phân giải cao (nhiều chi tiết hơn).
        • Resize sau grayscale → tránh nội suy màu không cần thiết.
        • HOG chỉ cần ảnh đúng kích thước ở bước cuối.

    Tại sao dùng LAB + CLAHE trên kênh L?
        → Tăng tương phản mà không làm sai lệch màu sắc (a, b giữ nguyên).

    Tại sao Bilateral Filter thay vì Gaussian?
        → Khử nhiễu nhưng bảo toàn cạnh (edge-preserving) – quan trọng cho HOG.
    """

    def __init__(self, target_size=(128, 128), clip_limit=1.5):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    def resize_reflect_padding(self, img: np.ndarray) -> np.ndarray:
        """
        Scale giữ tỉ lệ → fit vào target_size.
        Phần còn lại dùng BORDER_REFLECT_101 (gương) để tránh viền đen.
        """
        target_w, target_h = self.target_size
        h, w = img.shape[:2]

        if (w, h) == (target_w, target_h):
            return img

        scale = min(target_w / w, target_h / h)
        new_w = round(w * scale)
        new_h = round(h * scale)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top    = pad_h // 2;  bottom = pad_h - top
        left   = pad_w // 2;  right  = pad_w - left

        return cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_REFLECT_101
        )

    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Tăng cường ảnh trên độ phân giải GỐC (KHÔNG resize ở đây).
        Resize sẽ là bước CUỐI trong preprocess_image() sau khi đã grayscale.

        Các bước:
            ① Bilateral Filter  – khử nhiễu, giữ cạnh
            ② CLAHE trên kênh L – tăng tương phản cục bộ
        """
        if img_bgr is None:
            return None

        blurred = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=50, sigmaSpace=50)

        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return img_enhanced

COLOR_BINS_H = 16
COLOR_BINS_S = 8
COLOR_BINS_V = 8

enhancer = ImageEnhancer(target_size=(128, 128), clip_limit=1.5)

def preprocess_image(bgr):
    enhanced = enhancer.enhance(bgr)
    enhanced_color_resized = enhancer.resize_reflect_padding(enhanced)          # ảnh màu
    gray_resized = cv2.cvtColor(enhanced_color_resized, cv2.COLOR_BGR2GRAY)    # grayscale
    return gray_resized, enhanced_color_resized

def extract_color_histogram(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [COLOR_BINS_H], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [COLOR_BINS_S], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [COLOR_BINS_V], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h, norm_type=cv2.NORM_L1).flatten()
    hist_s = cv2.normalize(hist_s, hist_s, norm_type=cv2.NORM_L1).flatten()
    hist_v = cv2.normalize(hist_v, hist_v, norm_type=cv2.NORM_L1).flatten()
    return np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)

# Load model
with open('04_HOG_SVM/model.pkl', 'rb') as f:
    data = pickle.load(f)
clf = data["clf"]
le  = data["le"]

# Dự đoán
img = cv2.imread("/Users/quangminh/Documents/University/Computer vision/Project/Data/food101_24classes/omelette/1161524.jpg")
if img is None:
    raise FileNotFoundError("Không đọc được ảnh!")

gray, color_img = preprocess_image(img)

hog_feat, _ = hog(gray, orientations=9, pixels_per_cell=(8,8),
                  cells_per_block=(2,2), block_norm="L2-Hys",
                  visualize=True, feature_vector=True)

color_feat = extract_color_histogram(color_img)
feat = np.concatenate([hog_feat, color_feat]) 

proba = clf.predict_proba([feat])[0]
pred  = le.classes_[np.argmax(proba)]

print(f"Kết quả: {pred} ({proba.max():.1%} confidence)")
for i in np.argsort(proba)[::-1]:
    print(f"  {le.classes_[i]:<20} {proba[i]:.1%}")
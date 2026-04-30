"""
04_HOG_SVM/pipeline.py
=======================
Pipeline HOG + Color Histogram + SVM hoàn chỉnh – tích hợp ImageEnhancer từ 02_preprocessing.

Cấu trúc dataset yêu cầu:
    dataset/
    ├── apple/
    │   ├── img001.jpg
    │   └── ...
    ├── banana/
    └── orange/

Chạy:
    python 04_HOG_SVM/pipeline.py --dataset dataset/
    python 04_HOG_SVM/pipeline.py --dataset dataset/ --test path/anh_test.jpg

Pipeline tự động:
    ① Đọc dataset  →  ② Tiền xử lý (Bilateral→CLAHE→Gray→Resize)
    →  ③ HOG features  +  Color Histogram features (HSV)
    →  ④ Concatenate → Feature vector tổng hợp
    →  ⑤ Train SVM  →  ⑥ Đánh giá  →  ⑦ Lưu ảnh kết quả

Tại sao thêm Color Histogram?
    • HOG mô tả hình dạng / cấu trúc gradient (shape).
    • Color Histogram mô tả phân bố màu sắc (appearance).
    • Kết hợp cả hai → bổ sung thông tin, tăng khả năng phân biệt.
    • Ví dụ: chuối (vàng) vs táo (đỏ) → màu sắc là đặc trưng mạnh.
    
Tại sao dùng không gian màu HSV?
    • H (Hue) = màu thuần, ít bị ảnh hưởng bởi ánh sáng.
    • S (Saturation) = độ bão hoà, phân biệt màu sặc sỡ vs trung tính.
    • V (Value) = độ sáng, bổ sung thông tin cường độ sáng.
    • Tách H ra khỏi độ sáng → histogram màu bền hơn với điều kiện chiếu sáng.
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import exposure
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, accuracy_score)
import pickle

# ─── Thư mục lưu ảnh kết quả ─────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Tham số cố định ─────────────────────────────────────────
IMG_SIZE        = (128, 128)   # Resize tất cả ảnh về kích thước này
HOG_ORIENT      = 9            # Số bins histogram (0°–180°)
HOG_PPC         = (8, 8)       # Pixels per cell
HOG_CPB         = (2, 2)       # Cells per block

# ─── Tham số Color Histogram ─────────────────────────────────
COLOR_BINS_H    = 16           # Bins cho kênh Hue   (0–180 trong OpenCV)
COLOR_BINS_S    = 8            # Bins cho kênh Saturation (0–255)
COLOR_BINS_V    = 8            # Bins cho kênh Value      (0–255)
# Tổng: 16+8+8 = 32 bins → feature vector nhỏ gọn, đủ phân biệt màu

SUPPORTED_EXT   = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


# ════════════════════════════════════════════════════════════
#  IMAGE ENHANCER – tích hợp từ 02_preprocessing/preprocessing.py
# ════════════════════════════════════════════════════════════

class ImageEnhancer:
    """
    Pipeline enhance() (trên độ phân giải GỐC):
        Bilateral Filter  →  CLAHE trên kênh L (LAB)

    Resize tách ra khỏi enhance() và thực hiện CUỐI CÙNG ở preprocess_image()
    sau khi đã grayscale. Lý do:
        • Bilateral & CLAHE hiệu quả hơn ở độ phân giải cao.
        • Resize sau grayscale → tránh nội suy màu không cần thiết.
        • HOG chỉ cần ảnh đúng kích thước ở bước cuối.

    LƯU Ý với Color Histogram:
        Ảnh màu (BGR gốc sau enhance) cần được giữ lại riêng
        để trích xuất Color Histogram TRƯỚC khi grayscale.
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


# ── Khởi tạo enhancer dùng chung toàn pipeline ───────────────
enhancer = ImageEnhancer(target_size=IMG_SIZE, clip_limit=1.5)


# ════════════════════════════════════════════════════════════
#  BƯỚC 1 – ĐỌC DATASET
# ════════════════════════════════════════════════════════════

def load_dataset(dataset_path: str):
    """
    Đọc toàn bộ ảnh từ dataset_path.
    Mỗi thư mục con = 1 class.
    """
    if not os.path.isdir(dataset_path):
        sys.exit(f"[LỖI] Không tìm thấy thư mục dataset: {dataset_path}")

    classes = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
        and not d.startswith(".")
    ])

    if not classes:
        sys.exit(f"[LỖI] Không tìm thấy thư mục con (class) trong {dataset_path}")

    images, labels, paths = [], [], []
    print(f"\n{'─'*55}")
    print(f"  BƯỚC 1: ĐỌC DATASET từ '{dataset_path}'")
    print(f"{'─'*55}")
    print(f"  Tìm thấy {len(classes)} class: {classes}\n")

    for cls in classes:
        cls_dir   = os.path.join(dataset_path, cls)
        cls_files = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(SUPPORTED_EXT)
        ]
        if not cls_files:
            print(f"  ⚠  '{cls}' không có ảnh, bỏ qua")
            continue

        for fname in cls_files:
            fpath = os.path.join(cls_dir, fname)
            img   = cv2.imread(fpath)
            if img is None:
                continue
            images.append(img)
            labels.append(cls)
            paths.append(fpath)

        print(f"  ✓  {cls:<20} : {len(cls_files)} ảnh")

    print(f"\n  Tổng cộng: {len(images)} ảnh | {len(classes)} class")

    if len(images) == 0:
        sys.exit("[LỖI] Không đọc được ảnh nào. Kiểm tra lại định dạng file.")

    return images, labels, paths, classes


def visualize_samples(images, labels, classes, n_per_class=4):
    """Lưu ảnh mẫu từng class để kiểm tra dataset."""
    fig, axes = plt.subplots(
        len(classes), n_per_class,
        figsize=(n_per_class * 3, len(classes) * 3)
    )
    if len(classes) == 1:
        axes = [axes]

    fig.suptitle("Mẫu dataset – mỗi hàng là 1 class", fontsize=13, fontweight="bold")

    for i, cls in enumerate(classes):
        cls_imgs = [img for img, lbl in zip(images, labels) if lbl == cls][:n_per_class]
        for j in range(n_per_class):
            ax = axes[i][j] if n_per_class > 1 else axes[i]
            if j < len(cls_imgs):
                ax.imshow(cv2.cvtColor(cls_imgs[j], cv2.COLOR_BGR2RGB))
                if j == 0:
                    ax.set_ylabel(cls, fontsize=11, fontweight="bold")
            else:
                ax.axis("off")
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "01_dataset_samples.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Lưu: {out}")


# ════════════════════════════════════════════════════════════
#  BƯỚC 2 – TIỀN XỬ LÝ  (dùng ImageEnhancer)
# ════════════════════════════════════════════════════════════

def preprocess_image(bgr: np.ndarray):
    """
    Pipeline tiền xử lý cho HOG + Color Histogram:

        ① Bilateral Filter          – khử nhiễu, bảo toàn cạnh (độ phân giải gốc)
        ② CLAHE trên kênh L (LAB)   – tăng tương phản cục bộ  (độ phân giải gốc)
        ③ Resize + reflect padding  – co ảnh màu về IMG_SIZE  (cho Color Histogram)
        ④ Chuyển sang Grayscale     – chuẩn bị đầu vào HOG

    Lý do giữ ảnh màu (enhanced_resized_bgr):
        Color Histogram cần kênh màu → lấy từ ảnh đã enhance & resize.
        Thứ tự: enhance → resize → grayscale (khác pipeline HOG thuần).
        Resize TRƯỚC grayscale ở đây vì Color Histogram cần ảnh màu resize đúng kích thước.

    Returns:
        gray_resized        : ảnh grayscale uint8 kích thước IMG_SIZE  (cho HOG)
        enhanced_color_resized : ảnh BGR uint8 kích thước IMG_SIZE     (cho Color Histogram)
    """
    # ①② Tăng cường trên độ phân giải gốc
    enhanced_bgr = enhancer.enhance(bgr)

    # ③ Resize ảnh màu về IMG_SIZE (dùng cho Color Histogram)
    enhanced_color_resized = enhancer.resize_reflect_padding(enhanced_bgr)

    # ④ Grayscale từ ảnh đã resize (đầu vào HOG)
    gray_resized = cv2.cvtColor(enhanced_color_resized, cv2.COLOR_BGR2GRAY)

    return gray_resized, enhanced_color_resized


def preprocess_all(images: list, labels: list):
    """Tiền xử lý toàn bộ dataset. Trả về (grays, color_imgs)."""
    print(f"\n{'─'*55}")
    print(f"  BƯỚC 2: TIỀN XỬ LÝ ({len(images)} ảnh)")
    print(f"{'─'*55}")
    print(f"  Bilateral Filter → CLAHE (LAB/L) → Resize+Padding → Grayscale")

    results = [preprocess_image(img) for img in images]
    grays       = [r[0] for r in results]
    color_imgs  = [r[1] for r in results]

    print(f"  ✓ Hoàn thành tiền xử lý")
    return grays, color_imgs


def visualize_preprocessing(images, grays, color_imgs, labels, classes):
    """
    So sánh từng bước pipeline tiền xử lý:
        Cột 1 : Ảnh gốc (BGR)
        Cột 2 : Sau Bilateral Filter
        Cột 3 : Sau CLAHE
        Cột 4 : Resize (màu) → dùng cho Color Histogram
        Cột 5 : Grayscale   → dùng cho HOG
    """
    n_cls = min(len(classes), 4)
    fig, axes = plt.subplots(n_cls, 5, figsize=(18, n_cls * 3.5))
    if n_cls == 1:
        axes = [axes]

    col_titles = [
        "① Gốc (BGR)",
        "② Bilateral Filter",
        "③ CLAHE (L-channel)",
        "④ Resize (màu) → Color Hist",
        "⑤ Grayscale → HOG",
    ]
    for j, t in enumerate(col_titles):
        axes[0][j].set_title(t, fontsize=9, fontweight="bold")

    for i, cls in enumerate(classes[:n_cls]):
        idx  = next(k for k, lbl in enumerate(labels) if lbl == cls)
        orig = images[idx]

        step2 = cv2.bilateralFilter(orig, d=5, sigmaColor=50, sigmaSpace=50)
        lab = cv2.cvtColor(step2, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enh = enhancer.clahe.apply(l)
        step3 = cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2BGR)

        step4 = color_imgs[idx]  # ảnh màu sau resize
        step5 = grays[idx]       # grayscale sau resize

        axes[i][0].imshow(cv2.cvtColor(orig,  cv2.COLOR_BGR2RGB))
        axes[i][1].imshow(cv2.cvtColor(step2, cv2.COLOR_BGR2RGB))
        axes[i][2].imshow(cv2.cvtColor(step3, cv2.COLOR_BGR2RGB))
        axes[i][3].imshow(cv2.cvtColor(step4, cv2.COLOR_BGR2RGB))
        axes[i][4].imshow(step5, cmap="gray")

        axes[i][0].set_ylabel(cls, fontsize=10, fontweight="bold")
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(
        "Pipeline Tiền Xử Lý – Ảnh màu & grayscale cho Color Hist + HOG",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "02_preprocessing_steps.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Lưu: {out}")


# ════════════════════════════════════════════════════════════
#  BƯỚC 3A – TRÍCH XUẤT HOG FEATURES
# ════════════════════════════════════════════════════════════

def extract_hog(gray: np.ndarray):
    """
    Trích xuất HOG feature vector từ 1 ảnh grayscale.

    HOG Pipeline:
        Gradient (Gx, Gy)  →  Magnitude + Direction
        →  Cell histogram (9 bins, 0°–180°)
        →  Block normalization L2-Hys (2×2 cells/block)
        →  Feature vector 1D

    Returns:
        features  : np.ndarray 1D – HOG descriptor
        hog_image : ảnh visualisation HOG
    """
    features, hog_image = hog(
        gray,
        orientations=HOG_ORIENT,
        pixels_per_cell=HOG_PPC,
        cells_per_block=HOG_CPB,
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )
    return features, hog_image


# ════════════════════════════════════════════════════════════
#  BƯỚC 3B – TRÍCH XUẤT COLOR HISTOGRAM FEATURES (MỚI)
# ════════════════════════════════════════════════════════════

def extract_color_histogram(bgr_img: np.ndarray) -> np.ndarray:
    """
    Trích xuất Color Histogram trong không gian HSV.

    Tại sao HSV thay vì BGR?
        • BGR trộn lẫn màu và độ sáng → không bền dưới thay đổi ánh sáng.
        • HSV tách Hue (màu thuần) khỏi Value (độ sáng).
        • Histogram trên H, S, V riêng lẻ → mô tả phân bố màu ổn định hơn.

    Pipeline:
        BGR → HSV
        → tính histogram riêng từng kênh H, S, V
        → normalize L1 từng kênh (tổng = 1, bất biến với kích thước ảnh)
        → concatenate → feature vector 1D

    Số chiều: COLOR_BINS_H + COLOR_BINS_S + COLOR_BINS_V = 16+8+8 = 32 dims

    Returns:
        hist_feat : np.ndarray 1D, dtype=float32, đã normalize
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # Kênh H: range [0, 180] trong OpenCV
    hist_h = cv2.calcHist([hsv], [0], None, [COLOR_BINS_H], [0, 180])
    # Kênh S: range [0, 256]
    hist_s = cv2.calcHist([hsv], [1], None, [COLOR_BINS_S], [0, 256])
    # Kênh V: range [0, 256]
    hist_v = cv2.calcHist([hsv], [2], None, [COLOR_BINS_V], [0, 256])

    # Normalize L1 (tổng mỗi histogram = 1) → bất biến kích thước ảnh
    hist_h = cv2.normalize(hist_h, hist_h, norm_type=cv2.NORM_L1).flatten()
    hist_s = cv2.normalize(hist_s, hist_s, norm_type=cv2.NORM_L1).flatten()
    hist_v = cv2.normalize(hist_v, hist_v, norm_type=cv2.NORM_L1).flatten()

    return np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)


# ════════════════════════════════════════════════════════════
#  BƯỚC 3C – KẾT HỢP HOG + COLOR HISTOGRAM
# ════════════════════════════════════════════════════════════

def extract_combined_features(gray: np.ndarray, color_img: np.ndarray):
    """
    Trích xuất và kết hợp HOG + Color Histogram thành 1 feature vector duy nhất.

    Sơ đồ:
        ảnh grayscale (128×128)   →  HOG()          →  vec_hog   (N dims)
        ảnh màu BGR   (128×128)   →  ColorHist()    →  vec_color (32 dims)
                                          ↓
                            np.concatenate([vec_hog, vec_color])
                                          ↓
                              feature vector tổng hợp (N+32 dims)
                                          ↓
                                     SVM classifier

    Tại sao concatenate (không weighted sum)?
        • Giữ nguyên thông tin của cả hai descriptor.
        • StandardScaler sẽ chuẩn hoá scale khác nhau của HOG và Color Hist.
        • SVM tự học trọng số cho từng feature qua quá trình tối ưu kernel.

    Returns:
        combined  : np.ndarray 1D – HOG features + Color Histogram features
        hog_image : ảnh visualisation HOG (dùng cho biểu đồ)
    """
    hog_feat, hog_image = extract_hog(gray)
    color_feat          = extract_color_histogram(color_img)
    combined = np.concatenate([hog_feat, color_feat])
    return combined, hog_image


def extract_all_features(grays: list, color_imgs: list):
    """Trích xuất HOG + Color Histogram cho toàn bộ dataset (Parallel)."""
    print(f"\n{'─'*55}")
    print(f"  BƯỚC 3: TRÍCH XUẤT HOG + COLOR HISTOGRAM FEATURES")
    print(f"{'─'*55}")
    print(f"  HOG  : orient={HOG_ORIENT} | ppc={HOG_PPC} | cpb={HOG_CPB}")
    print(f"  Color: H={COLOR_BINS_H} bins | S={COLOR_BINS_S} bins | V={COLOR_BINS_V} bins (HSV)")

    def process(gray, color_img):
        feat, _ = extract_combined_features(gray, color_img)
        return feat

    X = Parallel(n_jobs=-1)(
        delayed(process)(g, c)
        for g, c in tqdm(zip(grays, color_imgs), desc="  Đang extract", total=len(grays))
    )

    X = np.array(X)

    # Phân tích kích thước feature vector
    hog_sample, _   = extract_hog(grays[0])
    color_sample    = extract_color_histogram(color_imgs[0])
    n_hog           = len(hog_sample)
    n_color         = len(color_sample)

    print(f"\n  HOG features  : {n_hog} dims")
    print(f"  Color Hist    : {n_color} dims  "
          f"(H:{COLOR_BINS_H} + S:{COLOR_BINS_S} + V:{COLOR_BINS_V})")
    print(f"  Combined total: {X.shape[1]} dims")
    print(f"  ✓ Feature matrix: {X.shape[0]} mẫu × {X.shape[1]} features")

    return X, n_hog, n_color


def visualize_hog_and_color(grays, color_imgs, labels, classes):
    """
    Trực quan HOG + Color Histogram cho mỗi class.
    Mỗi hàng: [Ảnh grayscale] [HOG Vis] [Ảnh màu HSV] [Hist H] [Hist S] [Hist V]
    """
    n_cls = min(len(classes), 5)
    fig, axes = plt.subplots(n_cls, 6, figsize=(22, n_cls * 3.5))
    if n_cls == 1:
        axes = [axes]

    col_titles = [
        "Grayscale (HOG input)",
        "HOG Visualization",
        "Màu (Color Hist input)",
        "Hist – Hue (H)",
        "Hist – Saturation (S)",
        "Hist – Value (V)",
    ]
    for j, t in enumerate(col_titles):
        axes[0][j].set_title(t, fontsize=9, fontweight="bold")

    hue_color  = "#e74c3c"
    sat_color  = "#8e44ad"
    val_color  = "#2c3e50"

    for i, cls in enumerate(classes[:n_cls]):
        idx       = next(k for k, lbl in enumerate(labels) if lbl == cls)
        gray      = grays[idx]
        color_img = color_imgs[idx]
        hsv       = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

        _, hog_vis = extract_hog(gray)
        hog_rescaled = exposure.rescale_intensity(hog_vis, in_range=(0, 10))

        hist_h = cv2.calcHist([hsv], [0], None, [COLOR_BINS_H], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [COLOR_BINS_S], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [COLOR_BINS_V], [0, 256]).flatten()
        hist_h /= hist_h.sum() + 1e-7
        hist_s /= hist_s.sum() + 1e-7
        hist_v /= hist_v.sum() + 1e-7

        axes[i][0].imshow(gray, cmap="gray")
        axes[i][0].set_ylabel(cls, fontsize=10, fontweight="bold")
        axes[i][1].imshow(hog_rescaled, cmap="gray")
        axes[i][2].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))

        axes[i][3].bar(range(COLOR_BINS_H), hist_h, color=hue_color, width=0.85)
        axes[i][3].set_xlabel("Hue bin", fontsize=7)

        axes[i][4].bar(range(COLOR_BINS_S), hist_s, color=sat_color, width=0.85)
        axes[i][4].set_xlabel("Sat bin", fontsize=7)

        axes[i][5].bar(range(COLOR_BINS_V), hist_v, color=val_color, width=0.85)
        axes[i][5].set_xlabel("Val bin", fontsize=7)

        for ax in axes[i][:3]:
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes[i][3:]:
            ax.set_yticks([])

    plt.suptitle(
        "HOG + Color Histogram (HSV) – mỗi hàng là 1 class",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "03_hog_color_visualization.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Lưu: {out}")


def visualize_feature_composition(n_hog: int, n_color: int):
    """
    Pie chart & bar chart mô tả tỉ lệ đóng góp HOG vs Color Histogram
    trong feature vector tổng hợp.
    """
    total = n_hog + n_color
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Cấu trúc Feature Vector Tổng Hợp (HOG + Color Hist)",
                 fontsize=13, fontweight="bold")

    # Pie chart
    sizes   = [n_hog, n_color]
    labels  = [f"HOG\n{n_hog} dims\n({n_hog/total:.1%})",
               f"Color Hist (HSV)\n{n_color} dims\n({n_color/total:.1%})"]
    colors  = ["#3498db", "#e67e22"]
    axes[0].pie(sizes, labels=labels, colors=colors, autopct="",
                startangle=90, wedgeprops=dict(width=0.55))
    axes[0].set_title("Tỉ lệ số chiều", fontsize=11)

    # Bar chart – breakdown Color Hist
    color_labels = [f"Hue\n{COLOR_BINS_H}d", f"Sat\n{COLOR_BINS_S}d", f"Val\n{COLOR_BINS_V}d"]
    color_vals   = [COLOR_BINS_H, COLOR_BINS_S, COLOR_BINS_V]
    color_colors = ["#e74c3c", "#8e44ad", "#2c3e50"]
    bars = axes[1].bar(["HOG"] + color_labels,
                       [n_hog] + color_vals,
                       color=["#3498db"] + color_colors,
                       edgecolor="white", linewidth=0.8)
    axes[1].set_ylabel("Số chiều (dims)", fontsize=11)
    axes[1].set_title("Số chiều từng thành phần", fontsize=11)
    for bar, val in zip(bars, [n_hog] + color_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "03b_feature_composition.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Lưu: {out}")


# ════════════════════════════════════════════════════════════
#  BƯỚC 4 – TRAIN SVM
# ════════════════════════════════════════════════════════════

def train_svm(X: np.ndarray, y_encoded: np.ndarray, test_size: float):
    """
    Train SVM với HOG + Color Histogram features.

    Pipeline:
        StandardScaler  → chuẩn hoá feature (zero mean, unit variance)
                          Quan trọng: HOG (~0–0.2) và Color Hist (~0–1)
                          có scale khác nhau → scaler cân bằng lại.
        SVC(rbf)        → SVM kernel RBF

    Returns:
        clf        : Pipeline đã train
        X_test     : tập test
        y_test     : nhãn test (encoded)
        split_label: "80-20" hoặc "70-30"
    """
    train_pct   = int((1 - test_size) * 100)
    test_pct    = int(test_size * 100)
    split_label = f"{train_pct}-{test_pct}"

    print(f"\n{'─'*55}")
    print(f"  TRAIN SVM – Split {train_pct}/{test_pct}")
    print(f"{'─'*55}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=42,
        stratify=y_encoded,
    )
    print(f"  Train: {len(X_train)} mẫu | Test: {len(X_test)} mẫu")

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, random_state=42)),
    ])

    clf.fit(X_train, y_train)
    print(f"  ✓ Train xong")

    return clf, X_test, y_test, split_label


# ════════════════════════════════════════════════════════════
#  BƯỚC 5 – ĐÁNH GIÁ
# ════════════════════════════════════════════════════════════

def evaluate(clf, X_test, y_test, le: LabelEncoder, split_label: str):
    """
    Đánh giá model và lưu biểu đồ – mỗi split lưu file riêng.
    """
    print(f"\n{'─'*55}")
    print(f"  ĐÁNH GIÁ – Split {split_label.replace('-', '/')}")
    print(f"{'─'*55}")

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {acc:.2%}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Confusion matrix ─────────────────────────────────────
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(le.classes_)),
                                    max(5, len(le.classes_) - 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix [{split_label.replace('-','/')}] – Accuracy {acc:.2%}",
                 fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"04_confusion_matrix_{split_label}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Lưu: {out}")

    # ── Biểu đồ accuracy mỗi class ──────────────────────────
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(max(7, len(le.classes_) * 1.2), 4))
    bars = ax.bar(le.classes_, per_class_acc * 100,
                  color=["#2ecc71" if a >= 0.7 else "#e74c3c" for a in per_class_acc],
                  edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(f"Accuracy từng class [{split_label.replace('-','/')}]",
                 fontsize=12, fontweight="bold")
    ax.axhline(70, color="gray", linestyle="--", linewidth=1, label="Ngưỡng 70%")
    ax.legend()
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0%}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, f"05_per_class_accuracy_{split_label}.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Lưu: {out2}")

    return acc, per_class_acc


def compare_splits(results: list, le: LabelEncoder):
    """
    So sánh accuracy tổng và từng class giữa 2 split (80/20 vs 70/30).
    """
    labels_split  = [r["split_label"].replace("-", "/") for r in results]
    accs          = [r["acc"] for r in results]
    classes       = le.classes_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("So sánh 80/20 vs 70/30 – HOG + Color Histogram + SVM",
                 fontsize=14, fontweight="bold")

    colors = ["#3498db", "#e67e22"]
    bars = axes[0].bar(labels_split, [a * 100 for a in accs],
                       color=colors, edgecolor="white", linewidth=0.8, width=0.4)
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].set_title("Accuracy tổng thể", fontsize=12, fontweight="bold")
    axes[0].axhline(70, color="gray", linestyle="--", linewidth=1, label="Ngưỡng 70%")
    axes[0].legend()
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f"{val:.2%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    x      = np.arange(len(classes))
    width  = 0.35
    for idx, (r, color) in enumerate(zip(results, colors)):
        offset = (idx - 0.5) * width
        bars2  = axes[1].bar(x + offset, r["per_class_acc"] * 100,
                              width, label=r["split_label"].replace("-", "/"),
                              color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars2, r["per_class_acc"]):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                         f"{val:.0%}", ha="center", va="bottom", fontsize=7)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=30, ha="right")
    axes[1].set_ylim(0, 115)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].set_title("Accuracy từng class", fontsize=12, fontweight="bold")
    axes[1].axhline(70, color="gray", linestyle="--", linewidth=1)
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "06_compare_splits.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Lưu: {out}")


# ════════════════════════════════════════════════════════════
#  BƯỚC 6 – DỰ ĐOÁN ẢNH MỚI
# ════════════════════════════════════════════════════════════

def predict_single(clf, le: LabelEncoder, img_path: str):
    """
    Dự đoán class của 1 ảnh mới.
    Pipeline nhất quán với lúc train: Bilateral → CLAHE → Resize → Gray + Color → HOG + Hist → SVM.
    """
    print(f"\n{'─'*55}")
    print(f"  DỰ ĐOÁN ẢNH: {img_path}")
    print(f"{'─'*55}")

    img = cv2.imread(img_path)
    if img is None:
        print(f"  [LỖI] Không đọc được ảnh: {img_path}")
        return

    gray, color_img = preprocess_image(img)
    feat, hog_vis   = extract_combined_features(gray, color_img)
    proba           = clf.predict_proba([feat])[0]
    pred            = le.classes_[np.argmax(proba)]

    hog_rescaled = exposure.rescale_intensity(hog_vis, in_range=(0, 10))

    # Color hist để hiển thị
    hsv    = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [COLOR_BINS_H], [0, 180]).flatten()
    hist_h /= hist_h.sum() + 1e-7

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Ảnh gốc", fontsize=11)

    axes[1].imshow(hog_rescaled, cmap="gray")
    axes[1].set_title("HOG Visualization", fontsize=11)

    axes[2].bar(range(COLOR_BINS_H), hist_h, color="#e74c3c", width=0.85)
    axes[2].set_title("Color Histogram (Hue)", fontsize=11)
    axes[2].set_xlabel("Hue bin")

    sorted_idx = np.argsort(proba)[::-1]
    bar_colors = ["#2ecc71" if le.classes_[i] == pred else "#95a5a6" for i in sorted_idx]
    axes[3].barh([le.classes_[i] for i in sorted_idx],
                 [proba[i] * 100 for i in sorted_idx],
                 color=bar_colors)
    axes[3].set_xlabel("Confidence (%)")
    axes[3].set_title(f"Dự đoán: {pred} ({proba.max():.1%})",
                      fontsize=11, fontweight="bold", color="#27ae60")
    axes[3].set_xlim(0, 110)

    for ax in axes[:2]:
        ax.axis("off")
    plt.suptitle("Kết quả phân loại HOG + Color Histogram + SVM",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "07_prediction_result.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Kết quả: {pred.upper()} ({proba.max():.1%} confidence)")
    print("  Top dự đoán:")
    for i in sorted_idx[:3]:
        print(f"    {le.classes_[i]:<20} {proba[i]:.1%}")
    print(f"  → Lưu: {out}")


def save_model(clf, le, model_path="04_HOG_SVM/model.pkl"):
    """Lưu model đã train để dùng lại."""
    with open(model_path, "wb") as f:
        pickle.dump({"clf": clf, "le": le}, f)
    print(f"\n  ✓ Đã lưu model: {model_path}")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline HOG + Color Histogram + SVM",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="dataset",
                        help="Đường dẫn thư mục dataset\n"
                             "VD: --dataset samples/  hoặc  --dataset ../dataset/")
    parser.add_argument("--test", type=str, default=None,
                        help="(Tùy chọn) Đường dẫn ảnh muốn dự đoán sau khi train\n"
                             "VD: --test samples/test.jpg")
    args = parser.parse_args()

    print("\n" + "═" * 55)
    print("   HOG + COLOR HISTOGRAM + SVM PIPELINE")
    print("   Tiền xử lý: Bilateral → CLAHE-LAB → Resize → HOG + HSV Hist")
    print("═" * 55)

    # ── 1. Đọc dataset ─────────────────────────────────────
    images, labels, paths, classes = load_dataset(args.dataset)
    visualize_samples(images, labels, classes)

    # ── 2. Tiền xử lý ──────────────────────────────────────
    grays, color_imgs = preprocess_all(images, labels)
    visualize_preprocessing(images, grays, color_imgs, labels, classes)

    # ── 3. HOG + Color Histogram features ──────────────────
    X, n_hog, n_color = extract_all_features(grays, color_imgs)
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    visualize_hog_and_color(grays, color_imgs, labels, classes)
    visualize_feature_composition(n_hog, n_color)

    # ── 4 & 5. Train + Đánh giá cho 2 tỉ lệ split ─────────
    print(f"\n{'═'*55}")
    print(f"  BƯỚC 4–5: TRAIN & ĐÁNH GIÁ (80/20 và 70/30)")
    print(f"{'═'*55}")

    results = []

    for test_size in [0.2, 0.3]:
        clf, X_test, y_test, split_label = train_svm(X, y, test_size)
        acc, per_class_acc = evaluate(clf, X_test, y_test, le, split_label)
        results.append({
            "split_label"   : split_label,
            "acc"           : acc,
            "per_class_acc" : per_class_acc,
            "clf"           : clf,
        })

    # ── So sánh 2 split ────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  SO SÁNH KẾT QUẢ")
    print(f"{'─'*55}")
    for r in results:
        print(f"  Split {r['split_label'].replace('-','/'):<6} → Accuracy: {r['acc']:.2%}")
    compare_splits(results, le)

    best     = max(results, key=lambda r: r["acc"])
    best_clf = best["clf"]
    print(f"\n  ✓ Model tốt nhất: Split {best['split_label'].replace('-','/')} "
          f"({best['acc']:.2%})")

    save_model(best_clf, le)

    # ── 6. Dự đoán ảnh mới ─────────────────────────────────
    if args.test:
        predict_single(best_clf, le, args.test)

    # ── Tổng kết ───────────────────────────────────────────
    print(f"\n{'═' * 55}")
    print(f"  ✅ PIPELINE HOÀN THÀNH")
    print(f"{'═' * 55}")
    print(f"  Ảnh kết quả lưu trong: 04_HOG_SVM/images/")
    print(f"")
    print(f"  01_dataset_samples.png          – Mẫu ảnh mỗi class")
    print(f"  02_preprocessing_steps.png      – 5 bước tiền xử lý")
    print(f"  03_hog_color_visualization.png  – HOG + Color Hist từng class")
    print(f"  03b_feature_composition.png     – Cấu trúc feature vector")
    print(f"  04_confusion_matrix_80-20.png   – Confusion matrix split 80/20")
    print(f"  04_confusion_matrix_70-30.png   – Confusion matrix split 70/30")
    print(f"  05_per_class_accuracy_80-20.png – Accuracy từng class split 80/20")
    print(f"  05_per_class_accuracy_70-30.png – Accuracy từng class split 70/30")
    print(f"  06_compare_splits.png           – So sánh 80/20 vs 70/30")
    if args.test:
        print(f"  07_prediction_result.png        – Kết quả dự đoán ảnh test")
    print(f"")
    print(f"  Dự đoán ảnh mới:")
    print(f"  python 04_HOG_SVM/pipeline.py --dataset dataset/ --test anh.jpg")
    print(f"{'═' * 55}\n")


if __name__ == "__main__":
    main()
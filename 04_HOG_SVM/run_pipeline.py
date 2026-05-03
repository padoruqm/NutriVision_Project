"""
04_HOG_SVM/pipeline.py
=======================
Pipeline HOG + SVM hoàn chỉnh – tích hợp ImageEnhancer từ 02_preprocessing.

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
    ① Đọc dataset  →  ② Tiền xử lý (Bilateral→CLAHE→Gray→Resize)  →  ③ HOG features
    →  ④ Train SVM  →  ⑤ Đánh giá  →  ⑥ Lưu ảnh kết quả
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

# Thư mục lưu ảnh kết quả
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tham số cố định
IMG_SIZE        = (128, 128)   # Resize tất cả ảnh về kích thước này
HOG_ORIENT      = 9            # Số bins histogram (0°–180°)
HOG_PPC         = (8, 8)       # Pixels per cell
HOG_CPB         = (2, 2)       # Cells per block
SUPPORTED_EXT   = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


#  IMAGE ENHANCER – tích hợp từ 02_preprocessing/preprocessing.py
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
        # CLAHE: clipLimit chống khuếch đại nhiễu quá mức
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

        # Denoise – Bilateral Filter (trên ảnh gốc, giữ toàn bộ chi tiết)
        blurred = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=50, sigmaSpace=50)

        # CLAHE trên kênh L của LAB
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return img_enhanced


# Khởi tạo enhancer dùng chung toàn pipeline
enhancer = ImageEnhancer(target_size=IMG_SIZE, clip_limit=1.5)



#  BƯỚC 1 – ĐỌC DATASET
def load_dataset(dataset_path: str):
    """
    Đọc toàn bộ ảnh từ dataset_path.
    Mỗi thư mục con = 1 class.

    Returns:
        images  : list ảnh BGR gốc (chưa xử lý)
        labels  : list tên class tương ứng
        paths   : list đường dẫn file
        classes : list tên class (sorted)
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


#  BƯỚC 2 – TIỀN XỬ LÝ  (dùng ImageEnhancer)
def preprocess_image(bgr: np.ndarray) -> np.ndarray:
    """
    Pipeline tiền xử lý – resize là bước CUỐI để phù hợp với HOG:
        ① Bilateral Filter          – khử nhiễu, bảo toàn cạnh (ở độ phân giải gốc)
        ② CLAHE trên kênh L (LAB)   – tăng tương phản cục bộ  (ở độ phân giải gốc)
        ③ Chuyển sang Grayscale     – loại bỏ thông tin màu trước khi nội suy
        ④ Resize + reflect padding  – co về IMG_SIZE, đầu vào chuẩn cho HOG

    Lý do resize cuối:
        Bilateral & CLAHE hiệu quả hơn khi còn đủ pixel gốc.
        Grayscale trước resize → nội suy 1 kênh, nhanh & chính xác hơn.

    Returns:
        gray_resized : ảnh grayscale uint8 kích thước IMG_SIZE
    """
    # Tăng cường trên độ phân giải gốc
    enhanced_bgr = enhancer.enhance(bgr)

    # Grayscale
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

    # Resize + reflect padding – bước cuối, chuẩn hoá kích thước cho HOG
    gray_resized = enhancer.resize_reflect_padding(gray)

    return gray_resized


def preprocess_all(images: list, labels: list):
    """Tiền xử lý toàn bộ dataset."""
    print(f"\n{'─'*55}")
    print(f"  BƯỚC 2: TIỀN XỬ LÝ ({len(images)} ảnh)")
    print(f"{'─'*55}")
    print(f"  Bilateral Filter → CLAHE (LAB/L) → Grayscale → Resize+Padding")

    processed = [preprocess_image(img) for img in images]

    print(f"  ✓ Hoàn thành tiền xử lý")
    return processed


def visualize_preprocessing(images, processed, labels, classes):
    """
    So sánh từng bước pipeline tiền xử lý cho mỗi class (thứ tự mới):
        Cột 1 : Ảnh gốc (BGR, độ phân giải gốc)
        Cột 2 : Sau Bilateral Filter   (khử nhiễu – vẫn ở kích thước gốc)
        Cột 3 : Sau CLAHE             (tăng tương phản – vẫn màu, kích thước gốc)
        Cột 4 : Grayscale             (trước khi resize)
        Cột 5 : Resize + Padding      (128×128, đầu vào HOG)
    """
    n_cls = min(len(classes), 4)
    fig, axes = plt.subplots(n_cls, 5, figsize=(18, n_cls * 3.5))
    if n_cls == 1:
        axes = [axes]

    col_titles = [
        "Gốc (BGR)",
        "Bilateral Filter",
        "CLAHE (L-channel)",
        "Grayscale",
        "Resize+Padding → HOG",
    ]
    for j, t in enumerate(col_titles):
        axes[0][j].set_title(t, fontsize=9, fontweight="bold")

    for i, cls in enumerate(classes[:n_cls]):
        idx  = next(k for k, lbl in enumerate(labels) if lbl == cls)
        orig = images[idx]

        # Bilateral trên ảnh gốc
        step2 = cv2.bilateralFilter(orig, d=5, sigmaColor=50, sigmaSpace=50)

        # CLAHE trên kênh L (vẫn ở kích thước gốc)
        lab = cv2.cvtColor(step2, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enh = enhancer.clahe.apply(l)
        step3 = cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2BGR)

        # Grayscale (trước khi resize)
        step4 = cv2.cvtColor(step3, cv2.COLOR_BGR2GRAY)

        # Resize + reflect padding – bước cuối
        step5 = processed[idx]   # đã tính sẵn trong preprocess_image()

        axes[i][0].imshow(cv2.cvtColor(orig,  cv2.COLOR_BGR2RGB))
        axes[i][1].imshow(cv2.cvtColor(step2, cv2.COLOR_BGR2RGB))
        axes[i][2].imshow(cv2.cvtColor(step3, cv2.COLOR_BGR2RGB))
        axes[i][3].imshow(step4, cmap="gray")
        axes[i][4].imshow(step5, cmap="gray")

        axes[i][0].set_ylabel(cls, fontsize=10, fontweight="bold")
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(
        "Pipeline Tiền Xử Lý – Resize là bước cuối (cho HOG)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "02_preprocessing_steps.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Lưu: {out}")


#  BƯỚC 3 – TRÍCH XUẤT HOG FEATURES
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


def extract_all_features(processed: list):
    """Trích xuất HOG cho toàn bộ dataset (Parallel)."""
    print(f"\n{'─'*55}")
    print(f"  BƯỚC 3: TRÍCH XUẤT HOG FEATURES (Parallel n_jobs=-1)")
    print(f"{'─'*55}")

    def process(gray):
        feat, _ = extract_hog(gray)
        return feat

    X = Parallel(n_jobs=-1)(
        delayed(process)(gray) for gray in tqdm(processed, desc="  Đang extract HOG")
    )

    X = np.array(X)

    print(f"\n  Tham số: orient={HOG_ORIENT} | ppc={HOG_PPC} | cpb={HOG_CPB}")
    print(f"  ✓ Feature matrix: {X.shape[0]} mẫu × {X.shape[1]} features")

    return X


def visualize_hog(processed, labels, classes):
    """Trực quan HOG cho 1 ảnh mỗi class."""
    n_cls = min(len(classes), 5)
    fig, axes = plt.subplots(n_cls, 3, figsize=(12, n_cls * 3.5))
    if n_cls == 1:
        axes = [axes]

    for j, t in enumerate(["Ảnh đã xử lý (Gray)", "HOG Visualization", "Histogram (128 bins đầu)"]):
        axes[0][j].set_title(t, fontsize=10, fontweight="bold")

    for i, cls in enumerate(classes[:n_cls]):
        idx  = next(k for k, lbl in enumerate(labels) if lbl == cls)
        gray = processed[idx]
        feat, hog_vis = extract_hog(gray)
        hog_rescaled  = exposure.rescale_intensity(hog_vis, in_range=(0, 10))

        axes[i][0].imshow(gray, cmap="gray")
        axes[i][0].set_ylabel(cls, fontsize=10, fontweight="bold")
        axes[i][1].imshow(hog_rescaled, cmap="gray")
        axes[i][2].bar(range(128), feat[:128], color="steelblue", width=1.0)
        for ax in axes[i][:2]:
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("HOG Features – mỗi hàng là 1 class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "03_hog_visualization.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Lưu: {out}")


#  BƯỚC 4 – TRAIN SVM
def train_svm(X: np.ndarray, y_encoded: np.ndarray, test_size: float):
    """
    Train SVM với HOG features cho 1 tỉ lệ split cụ thể.

    Pipeline:
        StandardScaler  → chuẩn hoá feature (zero mean, unit variance)
        SVC(rbf)        → SVM kernel RBF

    Args:
        test_size : 0.2 → split 80/20 | 0.3 → split 70/30

    Returns:
        clf        : Pipeline đã train
        X_test     : tập test
        y_test     : nhãn test (encoded)
        split_label: chuỗi "80/20" hoặc "70/30" dùng để đặt tên file
    """
    train_pct = int((1 - test_size) * 100)
    test_pct  = int(test_size * 100)
    split_label = f"{train_pct}-{test_pct}"   # "80-20" hoặc "70-30"

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


#  BƯỚC 5 – ĐÁNH GIÁ
def evaluate(clf, X_test, y_test, le: LabelEncoder, split_label: str):
    """
    Đánh giá model và lưu biểu đồ – mỗi split lưu file riêng.

    Args:
        split_label : "80-20" hoặc "70-30" – dùng để đặt tên file
    Returns:
        acc          : accuracy tổng trên tập test
        per_class_acc: accuracy từng class (np.ndarray)
    """
    print(f"\n{'─'*55}")
    print(f"  ĐÁNH GIÁ – Split {split_label.replace('-', '/')}")
    print(f"{'─'*55}")

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {acc:.2%}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
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

    # Biểu đồ accuracy mỗi class
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
    Vẽ biểu đồ so sánh accuracy tổng và từng class giữa 2 split (80/20 vs 70/30).

    Args:
        results : list of dict, mỗi phần tử chứa split_label, acc, per_class_acc
    """
    labels_split  = [r["split_label"].replace("-", "/") for r in results]
    accs          = [r["acc"] for r in results]
    per_class_all = [r["per_class_acc"] for r in results]
    classes       = le.classes_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("So sánh 80/20 vs 70/30", fontsize=14, fontweight="bold")

    # Cột trái: Accuracy tổng
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

    # Cột phải: Accuracy từng class (grouped bar)
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




#  BƯỚC 6 – DỰ ĐOÁN ẢNH MỚI
def predict_single(clf, le: LabelEncoder, img_path: str):
    """
    Dự đoán class của 1 ảnh mới.
    Pipeline: Đọc → preprocess_image() → HOG → SVM.predict
    Dùng cùng pipeline với lúc train: Bilateral → CLAHE → Gray → Resize.
    """
    print(f"\n{'─'*55}")
    print(f"  DỰ ĐOÁN ẢNH: {img_path}")
    print(f"{'─'*55}")

    img = cv2.imread(img_path)
    if img is None:
        print(f"  [LỖI] Không đọc được ảnh: {img_path}")
        return

    # Pipeline nhất quán với lúc train
    gray  = preprocess_image(img)
    feat, hog_vis = extract_hog(gray)
    proba = clf.predict_proba([feat])[0]
    pred  = le.classes_[np.argmax(proba)]

    hog_rescaled = exposure.rescale_intensity(hog_vis, in_range=(0, 10))

    # Visualise kết quả
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Ảnh gốc", fontsize=11)
    axes[1].imshow(hog_rescaled, cmap="gray")
    axes[1].set_title("HOG Visualization", fontsize=11)

    sorted_idx = np.argsort(proba)[::-1]
    colors = ["#2ecc71" if le.classes_[i] == pred else "#95a5a6" for i in sorted_idx]
    axes[2].barh([le.classes_[i] for i in sorted_idx],
                 [proba[i] * 100 for i in sorted_idx],
                 color=colors)
    axes[2].set_xlabel("Confidence (%)")
    axes[2].set_title(f"Dự đoán: {pred} ({proba.max():.1%})",
                      fontsize=11, fontweight="bold", color="#27ae60")
    axes[2].set_xlim(0, 110)

    for ax in axes[:2]:
        ax.axis("off")
    plt.suptitle("Kết quả phân loại HOG + SVM", fontsize=13, fontweight="bold")
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


#  MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline HOG + SVM – tích hợp ImageEnhancer",
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
    print("   HOG + SVM PIPELINE – Nutrivision Midterm")
    print("   Tiền xử lý: Bilateral → CLAHE-LAB → Grayscale → Resize (cuối)")
    print("═" * 55)

    # 1. Đọc dataset
    images, labels, paths, classes = load_dataset(args.dataset)
    visualize_samples(images, labels, classes)

    # 2. Tiền xử lý
    processed = preprocess_all(images, labels)
    visualize_preprocessing(images, processed, labels, classes)

    # 3. HOG features 
    X  = extract_all_features(processed)
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    visualize_hog(processed, labels, classes)

    # 4 & 5. Train + Đánh giá cho 2 tỉ lệ split
    print(f"\n{'═'*55}")
    print(f"  BƯỚC 4–5: TRAIN & ĐÁNH GIÁ (80/20 và 70/30)")
    print(f"{'═'*55}")

    results  = []   # lưu kết quả để so sánh
    best_clf = None

    for test_size in [0.2, 0.3]:
        clf, X_test, y_test, split_label = train_svm(X, y, test_size)
        acc, per_class_acc = evaluate(clf, X_test, y_test, le, split_label)
        results.append({
            "split_label"   : split_label,
            "acc"           : acc,
            "per_class_acc" : per_class_acc,
            "clf"           : clf,
        })
        if best_clf is None or acc > results[0]["acc"]:
            best_clf = clf

    # So sánh 2 split
    print(f"\n{'─'*55}")
    print(f"  SO SÁNH KẾT QUẢ")
    print(f"{'─'*55}")
    for r in results:
        print(f"  Split {r['split_label'].replace('-','/'):<6} → Accuracy: {r['acc']:.2%}")
    compare_splits(results, le)

    # Chọn split tốt hơn làm best model
    best = max(results, key=lambda r: r["acc"])
    best_clf = best["clf"]
    print(f"\n  ✓ Model tốt nhất: Split {best['split_label'].replace('-','/')} "
          f"({best['acc']:.2%})")

    # Lưu best model
    save_model(best_clf, le)

    # 6. Dự đoán ảnh mới (nếu có)
    if args.test:
        predict_single(best_clf, le, args.test)

    # Tổng kết
    print(f"\n{'═' * 55}")
    print(f"  PIPELINE HOÀN THÀNH")
    print(f"{'═' * 55}")
    print(f"  Ảnh kết quả lưu trong: 04_HOG_SVM/images/")
    print(f"")
    print(f"  01_dataset_samples.png          – Mẫu ảnh mỗi class")
    print(f"  02_preprocessing_steps.png      – 5 bước tiền xử lý")
    print(f"  03_hog_visualization.png        – HOG features từng class")
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
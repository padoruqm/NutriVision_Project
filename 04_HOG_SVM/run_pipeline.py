"""
04_HOG_SVM/pipeline.py
=======================
Pipeline HOG + SVM hoàn chỉnh cho dataset Nutrivision.

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
    ① Đọc dataset  →  ② Tiền xử lý  →  ③ HOG features
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
matplotlib.use("Agg")          # Không cần display, lưu file luôn
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import exposure

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
SUPPORTED_EXT   = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


# ════════════════════════════════════════════════════════════
#  BƯỚC 1 – ĐỌC DATASET
# ════════════════════════════════════════════════════════════

def load_dataset(dataset_path: str):
    """
    Đọc toàn bộ ảnh từ dataset_path.
    Mỗi thư mục con = 1 class.

    Returns:
        images : list ảnh BGR gốc (chưa xử lý) để visualise
        labels : list tên class tương ứng
        paths  : list đường dẫn file
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
#  BƯỚC 2 – TIỀN XỬ LÝ
# ════════════════════════════════════════════════════════════

def preprocess_image(bgr: np.ndarray) -> np.ndarray:
    """
    Pipeline tiền xử lý chuẩn cho 1 ảnh:
      ① Resize về IMG_SIZE (128×128)
      ② Chuyển sang Grayscale
      ③ CLAHE – cân bằng histogram cục bộ (tăng tương phản)
      ④ Gaussian Blur nhẹ – khử nhiễu

    Returns:
        gray_processed : ảnh grayscale đã xử lý, dtype uint8
    """
    # ① Resize
    resized = cv2.resize(bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # ② Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # ③ CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(gray)

    # ④ Gaussian blur nhẹ khử nhiễu (kernel 3×3, sigma tự tính)
    denoised  = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return denoised


def preprocess_all(images: list, labels: list):
    """Tiền xử lý toàn bộ dataset, trả về ảnh gốc và ảnh đã xử lý song song."""
    print(f"\n{'─'*55}")
    print(f"  BƯỚC 2: TIỀN XỬ LÝ ({len(images)} ảnh)")
    print(f"{'─'*55}")
    print(f"  Resize → {IMG_SIZE} | Grayscale | CLAHE | Gaussian Blur")

    processed = [preprocess_image(img) for img in images]

    print(f"  ✓ Hoàn thành tiền xử lý")
    return processed


def visualize_preprocessing(images, processed, labels, classes, n=2):
    """So sánh ảnh gốc vs ảnh đã tiền xử lý cho mỗi class."""
    n_cls = min(len(classes), 4)
    fig, axes = plt.subplots(n_cls, 4, figsize=(14, n_cls * 3.5))
    if n_cls == 1:
        axes = [axes]

    col_titles = ["Gốc (BGR)", "Resize + Gray", "Sau CLAHE", "Sau Gaussian"]
    for j, t in enumerate(col_titles):
        axes[0][j].set_title(t, fontsize=10, fontweight="bold")

    for i, cls in enumerate(classes[:n_cls]):
        idx   = next(k for k, l in enumerate(labels) if l == cls)
        orig  = images[idx]
        proc  = processed[idx]
        resized_color = cv2.resize(orig, IMG_SIZE, interpolation=cv2.INTER_AREA)
        gray_only     = cv2.cvtColor(resized_color, cv2.COLOR_BGR2GRAY)
        clahe_only    = cv2.createCLAHE(2.0, (8,8)).apply(gray_only)

        axes[i][0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        axes[i][1].imshow(gray_only,  cmap="gray")
        axes[i][2].imshow(clahe_only, cmap="gray")
        axes[i][3].imshow(proc,       cmap="gray")
        axes[i][0].set_ylabel(cls, fontsize=10, fontweight="bold")
        for ax in axes[i]: ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("Pipeline Tiền Xử Lý – từng bước", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "02_preprocessing_steps.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Lưu: {out}")


# ════════════════════════════════════════════════════════════
#  BƯỚC 3 – TRÍCH XUẤT HOG FEATURES
# ════════════════════════════════════════════════════════════

def extract_hog(gray: np.ndarray):
    """
    Trích xuất HOG feature vector từ 1 ảnh grayscale.

    HOG Pipeline:
      Gradient (Gx, Gy) → Magnitude + Direction
      → Cell histogram (9 bins, 0°–180°)
      → Block normalization L2-Hys (2×2 cells/block)
      → Feature vector 1D

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
    """Trích xuất HOG cho toàn bộ dataset."""
    print(f"\n{'─'*55}")
    print(f"  BƯỚC 3: TRÍCH XUẤT HOG FEATURES")
    print(f"{'─'*55}")

    X = []
    for gray in processed:
        feat, _ = extract_hog(gray)
        X.append(feat)

    X = np.array(X)
    print(f"  Tham số: orient={HOG_ORIENT} | ppc={HOG_PPC} | cpb={HOG_CPB}")
    print(f"  ✓ Feature matrix: {X.shape[0]} mẫu × {X.shape[1]} features")
    return X


def visualize_hog(processed, labels, classes):
    """Trực quan HOG cho 1 ảnh mỗi class."""
    n_cls = min(len(classes), 5)
    fig, axes = plt.subplots(n_cls, 3, figsize=(12, n_cls * 3.5))
    if n_cls == 1:
        axes = [axes]

    for j, t in enumerate(["Ảnh đã xử lý", "HOG Visualization", "Histogram (128 bins)"]):
        axes[0][j].set_title(t, fontsize=10, fontweight="bold")

    for i, cls in enumerate(classes[:n_cls]):
        idx  = next(k for k, l in enumerate(labels) if l == cls)
        gray = processed[idx]
        feat, hog_vis = extract_hog(gray)
        hog_rescaled  = exposure.rescale_intensity(hog_vis, in_range=(0, 10))

        axes[i][0].imshow(gray, cmap="gray")
        axes[i][0].set_ylabel(cls, fontsize=10, fontweight="bold")
        axes[i][1].imshow(hog_rescaled, cmap="gray")
        axes[i][2].bar(range(128), feat[:128], color="steelblue", width=1.0)
        for ax in axes[i][:2]: ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("HOG Features – mỗi hàng là 1 class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "03_hog_visualization.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Lưu: {out}")


# ════════════════════════════════════════════════════════════
#  BƯỚC 4 – TRAIN SVM
# ════════════════════════════════════════════════════════════

def train_svm(X: np.ndarray, y_encoded: np.ndarray, le: LabelEncoder):
    """
    Train SVM với HOG features.

    Pipeline:
        StandardScaler  → chuẩn hoá feature (zero mean, unit variance)
        SVC(rbf)        → SVM kernel RBF, tự tìm best decision boundary

    Tại sao StandardScaler?
        HOG features có scale khác nhau → chuẩn hoá giúp SVM hội tụ nhanh.

    Returns:
        clf      : Pipeline đã train
        X_test   : tập test
        y_test   : nhãn test (encoded)
    """
    print(f"\n{'─'*55}")
    print(f"  BƯỚC 4: TRAIN HOG + SVM")
    print(f"{'─'*55}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)} (tỉ lệ 80/20)")

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, random_state=42)),
    ])

    clf.fit(X_train, y_train)
    print(f"  ✓ Train xong")

    # Cross-validation 5-fold
    cv = StratifiedKFold(n_splits=min(5, len(le.classes_)), shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y_encoded, cv=cv, scoring="accuracy")
    print(f"\n  5-Fold Cross Validation:")
    print(f"    Scores : {cv_scores.round(3)}")
    print(f"    Mean   : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return clf, X_test, y_test


# ════════════════════════════════════════════════════════════
#  BƯỚC 5 – ĐÁNH GIÁ
# ════════════════════════════════════════════════════════════

def evaluate(clf, X_test, y_test, le: LabelEncoder):
    """Đánh giá model và lưu các biểu đồ kết quả."""
    print(f"\n{'─'*55}")
    print(f"  BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH")
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
    ax.set_title(f"Confusion Matrix – Accuracy {acc:.2%}", fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "04_confusion_matrix.png")
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
    ax.set_title("Accuracy từng class", fontsize=12, fontweight="bold")
    ax.axhline(70, color="gray", linestyle="--", linewidth=1, label="Ngưỡng 70%")
    ax.legend()
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.0%}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "05_per_class_accuracy.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Lưu: {out2}")

    return acc


# ════════════════════════════════════════════════════════════
#  BƯỚC 6 – DỰ ĐOÁN ẢNH MỚI
# ════════════════════════════════════════════════════════════

def predict_single(clf, le: LabelEncoder, img_path: str):
    """
    Dự đoán class của 1 ảnh mới.
    Pipeline: Đọc → Tiền xử lý → HOG → SVM.predict
    """
    print(f"\n{'─'*55}")
    print(f"  DỰ ĐOÁN ẢNH: {img_path}")
    print(f"{'─'*55}")

    img = cv2.imread(img_path)
    if img is None:
        print(f"  [LỖI] Không đọc được ảnh: {img_path}")
        return

    # Pipeline
    gray  = preprocess_image(img)
    feat, hog_vis = extract_hog(gray)
    proba = clf.predict_proba([feat])[0]
    pred  = le.classes_[np.argmax(proba)]

    hog_rescaled = exposure.rescale_intensity(hog_vis, in_range=(0, 10))

    # Visualise kết quả
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(cv2.cvtColor(img,  cv2.COLOR_BGR2RGB))
    axes[0].set_title("Ảnh gốc", fontsize=11)
    axes[1].imshow(hog_rescaled, cmap="gray")
    axes[1].set_title("HOG Visualization", fontsize=11)

    sorted_idx = np.argsort(proba)[::-1]
    colors = ["#2ecc71" if le.classes_[i] == pred else "#95a5a6" for i in sorted_idx]
    axes[2].barh([le.classes_[i] for i in sorted_idx],
                 [proba[i]*100 for i in sorted_idx],
                 color=colors)
    axes[2].set_xlabel("Confidence (%)")
    axes[2].set_title(f"Dự đoán: {pred} ({proba.max():.1%})",
                      fontsize=11, fontweight="bold", color="#27ae60")
    axes[2].set_xlim(0, 110)

    for ax in axes[:2]: ax.axis("off")
    plt.suptitle("Kết quả phân loại HOG + SVM", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "06_prediction_result.png")
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
        description="Pipeline HOG + SVM cho Nutrivision",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="dataset",
                        help="Đường dẫn thư mục dataset\n"
                             "VD: --dataset samples/  hoặc  --dataset ../dataset/")
    parser.add_argument("--test", type=str, default=None,
                        help="(Tùy chọn) Đường dẫn ảnh muốn dự đoán sau khi train\n"
                             "VD: --test samples/test.jpg")
    args = parser.parse_args()

    print("\n" + "═"*55)
    print("   HOG + SVM PIPELINE – Nutrivision Midterm")
    print("═"*55)

    # ── 1. Đọc dataset ─────────────────────────────────────
    images, labels, paths, classes = load_dataset(args.dataset)
    visualize_samples(images, labels, classes)

    # ── 2. Tiền xử lý ──────────────────────────────────────
    processed = preprocess_all(images, labels)
    visualize_preprocessing(images, processed, labels, classes)

    # ── 3. HOG features ────────────────────────────────────
    X  = extract_all_features(processed)
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    visualize_hog(processed, labels, classes)

    # ── 4. Train SVM ───────────────────────────────────────
    clf, X_test, y_test = train_svm(X, y, le)

    # ── 5. Đánh giá ────────────────────────────────────────
    evaluate(clf, X_test, y_test, le)

    # ── Lưu model ──────────────────────────────────────────
    save_model(clf, le)

    # ── 6. Dự đoán ảnh mới (nếu có) ───────────────────────
    if args.test:
        predict_single(clf, le, args.test)

    # ── Tổng kết ───────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  ✅ PIPELINE HOÀN THÀNH")
    print(f"{'═'*55}")
    print(f"  Ảnh kết quả lưu trong: 04_HOG_SVM/images/")
    print(f"")
    print(f"  01_dataset_samples.png     – Mẫu ảnh mỗi class")
    print(f"  02_preprocessing_steps.png – So sánh các bước tiền xử lý")
    print(f"  03_hog_visualization.png   – HOG features từng class")
    print(f"  04_confusion_matrix.png    – Ma trận nhầm lẫn")
    print(f"  05_per_class_accuracy.png  – Accuracy từng class")
    if args.test:
        print(f"  06_prediction_result.png   – Kết quả dự đoán ảnh test")
    print(f"")
    print(f"  Dự đoán ảnh mới:")
    print(f"  python 04_HOG_SVM/pipeline.py --dataset dataset/ --test anh.jpg")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
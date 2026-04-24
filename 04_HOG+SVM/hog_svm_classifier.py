"""
HOG + SVM: Pipeline phân loại/phát hiện đối tượng đầy đủ.

PIPELINE TỔNG THỂ:
  ┌─────────────────────────────────────────────────────────┐
  │  TRAINING PHASE                                         │
  │  Ảnh train → Resize → Grayscale → HOG features         │
  │            → Feature matrix X, labels y → Train SVM    │
  │                                                         │
  │  TESTING / DETECTION PHASE                              │
  │  Ảnh test → Sliding Window → HOG → SVM.predict → NMS   │
  └─────────────────────────────────────────────────────────┘

FILE NÀY DÙNG DỮ LIỆU TỔ HỢP (synthetic) để chạy được mà không cần dataset.
Trong thực tế, thay bằng ảnh thật từ dataset (INRIA, MNIST, CIFAR...).

Chạy: python 04_segmentation_detection/detection/hog_svm_classifier.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from utils.image_utils import load_or_create


# ─────────────────────────────────────────────────────────────
# TẠO DỮ LIỆU SYNTHETIC (thay bằng dataset thật khi có)
# ─────────────────────────────────────────────────────────────

def generate_synthetic_dataset(n_samples: int = 200,
                                img_size: tuple = (64, 64)) -> tuple:
    """
    Tạo dataset synthetic 2 lớp:
      - Class 0: ảnh có hình tròn (circle)
      - Class 1: ảnh có hình chữ nhật (rectangle)

    Thêm nhiễu Gaussian để thực tế hơn.

    Returns:
        images : list ảnh grayscale
        labels : list nhãn 0/1
    """
    images, labels = [], []
    h, w = img_size

    for i in range(n_samples):
        img = np.random.randint(50, 150, (h, w), dtype=np.uint8)  # background noise
        img = cv2.GaussianBlur(img, (3, 3), 1)

        if i % 2 == 0:
            # Class 0: hình tròn
            cx = np.random.randint(15, w - 15)
            cy = np.random.randint(15, h - 15)
            r  = np.random.randint(8, 18)
            cv2.circle(img, (cx, cy), r, np.random.randint(180, 255), -1)
            labels.append(0)
        else:
            # Class 1: hình chữ nhật
            x1 = np.random.randint(5, w // 2 - 5)
            y1 = np.random.randint(5, h // 2 - 5)
            x2 = x1 + np.random.randint(10, 30)
            y2 = y1 + np.random.randint(10, 30)
            cv2.rectangle(img, (x1, y1), (min(x2, w-5), min(y2, h-5)),
                          np.random.randint(180, 255), -1)
            labels.append(1)

        # Thêm nhiễu Gaussian nhẹ
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append(img)

    return images, labels


def visualize_dataset(images, labels, n_show=8, save_dir="images"):
    """Hiển thị mẫu dataset."""
    fig, axes = plt.subplots(2, n_show // 2, figsize=(16, 6))
    class_names = ["Circle (class 0)", "Rectangle (class 1)"]

    idx0 = [i for i, l in enumerate(labels) if l == 0][:n_show//2]
    idx1 = [i for i, l in enumerate(labels) if l == 1][:n_show//2]

    for j, idx in enumerate(idx0):
        axes[0][j].imshow(images[idx], cmap="gray")
        axes[0][j].set_title(class_names[0], fontsize=9); axes[0][j].axis("off")

    for j, idx in enumerate(idx1):
        axes[1][j].imshow(images[idx], cmap="gray")
        axes[1][j].set_title(class_names[1], fontsize=9); axes[1][j].axis("off")

    plt.suptitle(f"Dataset Synthetic: {len(images)} ảnh, 2 lớp", fontsize=12)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/hog_dataset.png", dpi=150); plt.show()


# ─────────────────────────────────────────────────────────────
# TRÍCH XUẤT HOG FEATURES
# ─────────────────────────────────────────────────────────────

def extract_features_batch(images: list,
                            orientations: int = 9,
                            pixels_per_cell: tuple = (8, 8),
                            cells_per_block: tuple = (2, 2)) -> np.ndarray:
    """
    Trích xuất HOG feature vector cho toàn bộ dataset.

    Returns:
        X : numpy array shape (N, n_features)
    """
    features_list = []
    for img in images:
        f = hog(img,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                feature_vector=True,
                block_norm="L2-Hys")
        features_list.append(f)

    X = np.array(features_list)
    print(f"📊 Feature matrix: {X.shape} "
          f"({X.shape[0]} samples × {X.shape[1]} features)")
    return X


# ─────────────────────────────────────────────────────────────
# TRAIN SVM
# ─────────────────────────────────────────────────────────────

def train_hog_svm(X: np.ndarray, y: np.ndarray, save_dir="images") -> Pipeline:
    """
    Train SVM với HOG features.

    Pipeline:
        StandardScaler → chuẩn hoá feature (zero mean, unit variance)
        SVC(kernel='rbf') → SVM với kernel RBF

    Tại sao cần StandardScaler?
        HOG features có scale khác nhau → chuẩn hoá giúp SVM hội tụ tốt hơn.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    print(f"\n📦 Train: {len(X_train)} | Test: {len(X_test)}")

    # Pipeline: Scale → SVM
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=1.0, gamma="scale",
                       probability=True, random_state=42)),
    ])

    clf.fit(X_train, y_train)

    # Đánh giá
    y_pred = clf.predict(X_test)
    acc    = (y_pred == y_test).mean()

    print(f"\n🎯 Accuracy: {acc:.2%}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Circle", "Rectangle"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion_matrix(cm, ["Circle", "Rectangle"], save_dir)

    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"\n🔁 5-Fold Cross-Validation:")
    print(f"   Scores: {cv_scores.round(3)}")
    print(f"   Mean ± Std: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return clf


def _plot_confusion_matrix(cm, classes, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix – HOG + SVM")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=16, fontweight="bold")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/hog_svm_confusion_matrix.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
# SLIDING WINDOW DETECTION
# ─────────────────────────────────────────────────────────────

def sliding_window(img, win_size=(64,64), step=16):
    """Generator: trả về từng cửa sổ (y,x,window) khi trượt qua ảnh."""
    h, w = img.shape[:2]
    for y in range(0, h - win_size[1], step):
        for x in range(0, w - win_size[0], step):
            yield y, x, img[y:y+win_size[1], x:x+win_size[0]]


def non_maximum_suppression(boxes, scores, iou_threshold=0.4):
    """
    NMS – loại bỏ bounding boxes trùng nhau.
    Giữ lại box có score cao nhất, loại những box có IoU > threshold.

    IoU = Intersection over Union = diện tích giao / diện tích hợp
    """
    if len(boxes) == 0:
        return []

    boxes  = np.array(boxes)
    scores = np.array(scores)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []

    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        inds  = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def detect_with_sliding_window(clf: Pipeline, test_img: np.ndarray,
                                 win_size=(64,64), step=16,
                                 save_dir="images"):
    """
    Phát hiện đối tượng bằng Sliding Window + HOG + SVM + NMS.
    """
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) if test_img.ndim == 3 else test_img
    detections, scores = [], []

    for y, x, window in sliding_window(gray, win_size, step):
        if window.shape[:2] != win_size: continue
        f = hog(window, orientations=9, pixels_per_cell=(8,8),
                cells_per_block=(2,2), feature_vector=True, block_norm="L2-Hys")
        prob = clf.predict_proba([f])[0]
        if prob[1] > 0.75:          # Ngưỡng confidence cho class Rectangle
            detections.append((x, y, win_size[0], win_size[1]))
            scores.append(prob[1])

    # NMS
    keep = non_maximum_suppression(detections, scores)
    final_boxes = [detections[i] for i in keep]

    # Vẽ kết quả
    result_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in final_boxes:
        cv2.rectangle(result_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Test Image")
    # Tất cả detections trước NMS
    all_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in detections:
        cv2.rectangle(all_vis, (x,y),(x+w,y+h),(0,0,255),1)
    axes[1].imshow(cv2.cvtColor(all_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Trước NMS\n{len(detections)} detections")
    axes[2].imshow(cv2.cvtColor(result_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Sau NMS\n{len(final_boxes)} detections")
    for ax in axes: ax.axis("off")

    plt.suptitle("Sliding Window Detection + NMS", fontsize=13)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/hog_sliding_window.png", dpi=150)
    plt.show()
    print(f"✅ Detections: {len(detections)} → sau NMS: {len(final_boxes)}")


if __name__ == "__main__":
    save_dir = "images"

    print("=" * 60)
    print("  HOG + SVM CLASSIFIER – Pipeline đầy đủ")
    print("=" * 60)

    # 1. Tạo dataset
    print("\n[1/5] Tạo synthetic dataset...")
    images, labels = generate_synthetic_dataset(n_samples=300, img_size=(64,64))
    visualize_dataset(images, labels, save_dir=save_dir)

    # 2. Extract HOG features
    print("\n[2/5] Trích xuất HOG features...")
    X = extract_features_batch(images)
    y = np.array(labels)

    # 3. Train SVM
    print("\n[3/5] Train HOG + SVM...")
    clf = train_hog_svm(X, y, save_dir=save_dir)

    # 4. Test trên ảnh mới
    print("\n[4/5] Demo trên ảnh test đơn lẻ...")
    test_imgs, test_lbls = generate_synthetic_dataset(n_samples=6)
    class_names = ["Circle", "Rectangle"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (img, true_lbl) in zip(axes.flat, zip(test_imgs, test_lbls)):
        f = hog(img, orientations=9, pixels_per_cell=(8,8),
                cells_per_block=(2,2), feature_vector=True, block_norm="L2-Hys")
        pred  = clf.predict([f])[0]
        proba = clf.predict_proba([f])[0]
        color = "green" if pred == true_lbl else "red"
        ax.imshow(img, cmap="gray")
        ax.set_title(f"True: {class_names[true_lbl]}\n"
                     f"Pred: {class_names[pred]} ({proba[pred]:.0%})",
                     color=color, fontsize=9)
        ax.axis("off")
    plt.suptitle("Prediction trên ảnh test (xanh=đúng, đỏ=sai)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/hog_svm_predictions.png", dpi=150); plt.show()

    # 5. Sliding window
    print("\n[5/5] Demo Sliding Window Detection...")
    test_scene = np.random.randint(80, 140, (200, 300), dtype=np.uint8)
    # Thêm vài hình chữ nhật vào ảnh để detect
    cv2.rectangle(test_scene, (50,40),  (100,90),  220, -1)
    cv2.rectangle(test_scene, (170,100),(230,160), 220, -1)
    detect_with_sliding_window(clf, test_scene, save_dir=save_dir)

    print("\n✅ Tất cả kết quả lưu trong detection/images/")
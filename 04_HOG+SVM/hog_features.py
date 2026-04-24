"""
HOG (Histogram of Oriented Gradients)

PIPELINE HOG (5 bước):
  ① Gradient computation  : tính Gx, Gy → magnitude + direction
  ② Cell histogram        : mỗi cell 8×8px → histogram 9 bins (0°–180°)
  ③ Block normalization   : nhóm 2×2 cells → normalize L2 → bất biến ánh sáng
  ④ Feature vector        : ghép tất cả block → 1 vector dài
  ⑤ Classifier (SVM)      : SVM dùng feature vector để phân loại

TẠI SAO HOG HOẠT ĐỘNG?
  - Gradient nắm bắt cấu trúc hình dạng (edges, corners)
  - Bất biến với thay đổi nhỏ về màu sắc và ánh sáng (do normalize)
  - Histogram ổn định hơn so với dùng gradient trực tiếp

Chạy: python 04_segmentation_detection/detection/hog_features.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from utils.image_utils import load_or_create


# ─────────────────────────────────────────────────────────────
# BƯỚC 1: GRADIENT COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_gradients(gray: np.ndarray) -> tuple:
    """
    Tính gradient theo X và Y bằng bộ lọc [-1, 0, 1].
    OpenCV dùng kernel: Gx = [-1, 0, 1] ; Gy = [-1, 0, 1]ᵀ

    Returns:
        gx, gy      : gradient theo X và Y
        magnitude   : |∇f| = √(Gx² + Gy²)
        direction   : θ = arctan(Gy/Gx), lấy giá trị [0°, 180°]
    """
    gx        = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
    gy        = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.degrees(np.arctan2(np.abs(gy), np.abs(gx)))  # [0°, 90°]
    # Mở rộng ra [0°, 180°] theo convention HOG
    direction_full = np.degrees(np.arctan2(gy, gx)) % 180

    return gx, gy, magnitude, direction_full


def visualize_gradients(img: np.ndarray, save_dir="images"):
    """Trực quan hoá gradient – bước đầu tiên của HOG."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gx, gy, mag, direction = compute_gradients(gray)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("HOG – Bước 1: Gradient Computation", fontsize=14, fontweight="bold")

    data = [
        (gray,                          "Ảnh Grayscale",           "gray"),
        (np.abs(gx),                    "Gx (gradient ngang)",     "RdBu_r"),
        (np.abs(gy),                    "Gy (gradient dọc)",       "RdBu_r"),
        (mag,                           "Magnitude |∇f|",          "hot"),
        (direction,                     "Direction θ (0°–180°)",   "hsv"),
        (np.clip(mag, 0, 255).astype(np.uint8), "Magnitude (clipped)", "gray"),
    ]

    for ax, (d, title, cmap) in zip(axes.flat, data):
        im = ax.imshow(d, cmap=cmap)
        ax.set_title(title, fontsize=11); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/hog_step1_gradient.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
# BƯỚC 2 + 3: CELL HISTOGRAM & BLOCK NORMALIZATION
# ─────────────────────────────────────────────────────────────

def cell_histogram_demo(gray: np.ndarray, save_dir="images"):
    """
    Demo tính histogram gradient trong 1 cell 8×8.
    9 bins: [0°,20°), [20°,40°), ..., [160°,180°)
    """
    # Lấy vùng nhỏ (1 cell ví dụ)
    h, w = gray.shape
    cell = gray[h//2 - 4 : h//2 + 4, w//2 - 4 : w//2 + 4]

    gx    = cv2.Sobel(cell.astype(np.float64), cv2.CV_64F, 1, 0, ksize=1)
    gy    = cv2.Sobel(cell.astype(np.float64), cv2.CV_64F, 0, 1, ksize=1)
    mag_c = np.sqrt(gx**2 + gy**2)
    ang_c = np.degrees(np.arctan2(np.abs(gy), np.abs(gx)))

    bins   = np.arange(0, 180, 20)   # 9 bins: 0,20,40,...,160
    hist   = np.zeros(9)
    labels = [f"{b}°" for b in bins]

    # Tích luỹ magnitude vào bin tương ứng (soft assignment đơn giản)
    for r in range(8):
        for c in range(8):
            bin_idx = int(ang_c[r, c] // 20) % 9
            hist[bin_idx] += mag_c[r, c]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("HOG – Bước 2: Cell Histogram (1 ô 8×8)", fontsize=13)

    axes[0].imshow(cv2.resize(cell, (64, 64), interpolation=cv2.INTER_NEAREST), cmap="gray")
    axes[0].set_title("Cell 8×8 (phóng to)"); axes[0].axis("off")

    im = axes[1].imshow(mag_c, cmap="hot"); axes[1].set_title("Gradient magnitude")
    plt.colorbar(im, ax=axes[1])

    axes[2].bar(labels, hist, color="steelblue", edgecolor="navy")
    axes[2].set_title("Histogram 9 bins (0°–180°)")
    axes[2].set_xlabel("Bin (hướng gradient)"); axes[2].set_ylabel("Tổng magnitude")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/hog_step2_cell_hist.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────
# FULL HOG FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_hog_features(img: np.ndarray,
                          orientations: int = 9,
                          pixels_per_cell: tuple = (8, 8),
                          cells_per_block: tuple = (2, 2)) -> tuple:
    """
    Trích xuất HOG features bằng scikit-image.

    Args:
        orientations    : số bins histogram (thường 9)
        pixels_per_cell : kích thước 1 cell tính bằng pixel
        cells_per_block : số cell theo mỗi chiều trong 1 block

    Returns:
        features  : 1D numpy array – HOG descriptor
        hog_image : ảnh visualisation HOG
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    features, hog_image = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        feature_vector=True,
        block_norm="L2-Hys",
    )
    # Tăng độ tương phản để dễ nhìn
    hog_vis = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return features, hog_vis


def visualize_full_hog(img: np.ndarray, save_dir="images"):
    """So sánh ảnh gốc và HOG visualization."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    features, hog_vis = extract_hog_features(img)

    # Tính số chiều feature vector
    h, w = gray.shape
    n_cells_x = w // 8; n_cells_y = h // 8
    n_blocks_x = n_cells_x - 1; n_blocks_y = n_cells_y - 1
    n_features = n_blocks_x * n_blocks_y * 4 * 9  # blocks × 4cells/block × 9bins

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("HOG – Full Feature Extraction", fontsize=14, fontweight="bold")

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim==3 else gray,
                   cmap="gray" if img.ndim==2 else None)
    axes[0].set_title("Ảnh Gốc"); axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title(f"Grayscale\n{w}×{h} px"); axes[1].axis("off")

    axes[2].imshow(hog_vis, cmap="gray")
    axes[2].set_title(f"HOG Visualization\n{len(features):,} features")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/hog_visualization.png", dpi=150)
    plt.show()

    print(f"\n📊 HOG Feature Vector Info:")
    print(f"   Ảnh       : {w}×{h} px")
    print(f"   Cells     : {n_cells_x}×{n_cells_y} = {n_cells_x*n_cells_y} cells")
    print(f"   Blocks    : {n_blocks_x}×{n_blocks_y} = {n_blocks_x*n_blocks_y} blocks")
    print(f"   Features  : {len(features):,} chiều")
    print(f"   = {n_blocks_x*n_blocks_y} blocks × 4 cells/block × 9 bins = {n_features}")


# ─────────────────────────────────────────────────────────────
# SO SÁNH THAM SỐ
# ─────────────────────────────────────────────────────────────

def parameter_study(img: np.ndarray, save_dir="images"):
    """Ảnh hưởng của các tham số HOG lên kết quả."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    configs = [
        (9,  (4,  4),  (2, 2), "Fine\n4×4 cells"),
        (9,  (8,  8),  (2, 2), "Default\n8×8 cells"),
        (9,  (16, 16), (2, 2), "Coarse\n16×16 cells"),
        (4,  (8,  8),  (2, 2), "4 orientations"),
        (18, (8,  8),  (2, 2), "18 orientations"),
    ]

    fig, axes = plt.subplots(2, len(configs) + 1, figsize=(22, 9))
    fig.suptitle("Ảnh Hưởng Tham Số HOG", fontsize=14, fontweight="bold")

    axes[0][0].imshow(gray, cmap="gray"); axes[0][0].set_title("Gốc"); axes[0][0].axis("off")
    axes[1][0].axis("off")

    for j, (orient, ppc, cpb, label) in enumerate(configs, 1):
        try:
            feats, hog_vis = hog(gray, orientations=orient,
                                  pixels_per_cell=ppc, cells_per_block=cpb,
                                  visualize=True, feature_vector=True)
            hog_vis = exposure.rescale_intensity(hog_vis, in_range=(0, 10))
            axes[0][j].imshow(gray, cmap="gray")
            axes[0][j].set_title(label, fontsize=9); axes[0][j].axis("off")
            axes[1][j].imshow(hog_vis, cmap="gray")
            axes[1][j].set_title(f"{len(feats):,} features", fontsize=9)
            axes[1][j].axis("off")
        except Exception:
            axes[0][j].axis("off"); axes[1][j].axis("off")

    axes[0][0].set_ylabel("Ảnh", fontsize=11)
    axes[1][0].set_ylabel("HOG Vis", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/hog_parameters.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    img = load_or_create("../../samples/test.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("=== HOG: BƯỚC 1 – GRADIENT ===")
    visualize_gradients(img)

    print("\n=== HOG: BƯỚC 2 – CELL HISTOGRAM ===")
    cell_histogram_demo(gray)

    print("\n=== HOG: FULL VISUALIZATION ===")
    visualize_full_hog(img)

    print("\n=== HOG: SO SÁNH THAM SỐ ===")
    parameter_study(img)
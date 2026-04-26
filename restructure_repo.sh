#!/bin/bash
# ============================================================
#  restructure_repo.sh
#  Tái cấu trúc repo xử lý ảnh giữa kì
#  Chạy từ ROOT của repo: bash restructure_repo.sh
# ============================================================

set -e  # Dừng ngay nếu có lỗi

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo "========================================"
echo "  Tái cấu trúc repo xử lý ảnh giữa kì"
echo "========================================"
echo ""

# ── Kiểm tra đang đứng đúng vị trí ──────────────────────────
if [ ! -f "README.md" ]; then
  echo -e "${RED}✗ Không tìm thấy README.md${NC}"
  echo "  → Hãy chạy script này từ thư mục ROOT của repo!"
  exit 1
fi

echo -e "${GREEN}✓ Đang đứng tại: $(pwd)${NC}"
echo ""

# ════════════════════════════════════════════════════════════
# BƯỚC 1 – Tạo các thư mục còn thiếu
# ════════════════════════════════════════════════════════════
echo "[ 1/6 ] Tạo thư mục..."

mkdir -p 01_color_representation/images
mkdir -p 02_preprocessing/images
mkdir -p 03_morphology/images
mkdir -p 04_HOG_SVM/images      # Tên mới: dấu + → dấu _
mkdir -p utils
mkdir -p samples

echo -e "${GREEN}  ✓ Đã tạo đủ thư mục${NC}"

# ════════════════════════════════════════════════════════════
# BƯỚC 2 – Đổi tên 04_HOG+SVM → 04_HOG_SVM (nếu tồn tại)
# ════════════════════════════════════════════════════════════
echo ""
echo "[ 2/6 ] Đổi tên thư mục 04..."

OLD_DIR="04_HOG+SVM"
NEW_DIR="04_HOG_SVM"

if [ -d "$OLD_DIR" ]; then
  # Di chuyển từng file sang thư mục mới
  cp -r "$OLD_DIR"/. "$NEW_DIR"/
  rm -rf "$OLD_DIR"
  echo -e "${GREEN}  ✓ Đổi tên: '${OLD_DIR}' → '${NEW_DIR}'${NC}"

  # Thông báo git nếu đang trong repo
  if git rev-parse --git-dir > /dev/null 2>&1; then
    git rm -r --cached "$OLD_DIR" 2>/dev/null || true
    git add "$NEW_DIR" 2>/dev/null || true
    echo -e "${YELLOW}  ⚠ Đã cập nhật git index – nhớ commit sau!${NC}"
  fi
else
  echo -e "${YELLOW}  ⚠ '${OLD_DIR}' không tồn tại, bỏ qua bước này${NC}"
fi

# ════════════════════════════════════════════════════════════
# BƯỚC 3 – Tạo file Python còn thiếu (chỉ tạo nếu CHƯA có)
# ════════════════════════════════════════════════════════════
echo ""
echo "[ 3/6 ] Tạo file Python còn thiếu..."

create_if_missing() {
  local path="$1"
  local content="$2"
  if [ ! -f "$path" ]; then
    echo "$content" > "$path"
    echo -e "${GREEN}  + Tạo: ${path}${NC}"
  else
    echo -e "${YELLOW}  ~ Đã có: ${path} (giữ nguyên)${NC}"
  fi
}

# ── 01_color_representation ──────────────────────────────────
create_if_missing "01_color_representation/color_spaces.py" \
'"""
color_spaces.py – Chủ đề 1
============================
Chuyển đổi BGR ↔ RGB / HSV / LAB / YCrCb / Grayscale
Lọc màu bằng HSV (ứng dụng thực tế)

Chạy: python 01_color_representation/color_spaces.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cv2, numpy as np, matplotlib.pyplot as plt
from utils.image_utils import load_or_create

def convert_all(bgr):
    return {
        "BGR (gốc)":  bgr,
        "RGB":        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        "Grayscale":  cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
        "HSV":        cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
        "LAB":        cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB),
        "YCrCb":      cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb),
    }

def plot_color_spaces(bgr, save_dir="images"):
    os.makedirs(save_dir, exist_ok=True)
    conv = convert_all(bgr)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("So sánh các Hệ Màu", fontsize=16, fontweight="bold")
    for ax, (name, img) in zip(axes.flat, conv.items()):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if name == "BGR (gốc)" else img)
        ax.set_title(name, fontsize=12); ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/color_spaces.png", dpi=150); plt.show()

def plot_hsv_channels(bgr, save_dir="images"):
    """Tách H, S, V – quan trọng khi lọc màu theo HSV."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Phân tích kênh HSV", fontsize=13)
    for ax, (img, title, cmap) in zip(axes, [
        (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "Ảnh gốc", None),
        (h, "H – Hue (Màu sắc)", "hsv"),
        (s, "S – Saturation",    "gray"),
        (v, "V – Value (Sáng)",  "gray"),
    ]):
        ax.imshow(img, cmap=cmap); ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/hsv_channels.png", dpi=150); plt.show()

if __name__ == "__main__":
    img = load_or_create("../samples/test.jpg")
    print(f"Kích thước: {img.shape} | dtype: {img.dtype}")
    print("Lưu ý: OpenCV đọc theo BGR, KHÔNG phải RGB!")
    plot_color_spaces(img)
    plot_hsv_channels(img)
'

create_if_missing "01_color_representation/histogram.py" \
'"""
histogram.py – Chủ đề 1
=========================
Histogram ảnh + Histogram Equalization + CLAHE

Chạy: python 01_color_representation/histogram.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cv2, numpy as np, matplotlib.pyplot as plt
from utils.image_utils import load_or_create

def histogram_equalization(gray, save_dir="images"):
    """Global EQ vs CLAHE (cân bằng cục bộ – tốt hơn)."""
    eq    = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, img, title in zip(axes[0], [gray, eq, clahe],
                               ["Gốc", "Global EQ", "CLAHE"]):
        ax.imshow(img, cmap="gray"); ax.set_title(title); ax.axis("off")
    for ax, img, color in zip(axes[1], [gray, eq, clahe], ["black","blue","red"]):
        h = cv2.calcHist([img],[0],None,[256],[0,256])
        ax.plot(h, color=color); ax.fill_between(range(256), h.flatten(), alpha=0.25, color=color)
        ax.set_xlim([0,256])
    plt.suptitle("Histogram Equalization – Tăng độ tương phản", fontsize=13)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/histogram_eq.png", dpi=150); plt.show()

if __name__ == "__main__":
    img  = load_or_create("../samples/test.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    histogram_equalization(gray)
'

# ── 02_preprocessing ─────────────────────────────────────────
create_if_missing "02_preprocessing/pixel_operations.py" \
'"""
pixel_operations.py – Chủ đề 2
=================================
Thao tác điểm ảnh: sáng, tương phản, gamma, threshold.
Công thức tổng quát: g(x,y) = T[f(x,y)]

Chạy: python 02_preprocessing/pixel_operations.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cv2, numpy as np, matplotlib.pyplot as plt
from utils.image_utils import load_or_create

def brightness(img, beta):  return np.clip(img.astype(np.int16)+beta, 0,255).astype(np.uint8)
def contrast(img, alpha):   return np.clip(img.astype(np.float32)*alpha, 0,255).astype(np.uint8)
def gamma(img, g):
    lut = np.array([((i/255.0)**(1/g))*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)

def demo(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, t_fixed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, t_otsu  = cv2.threshold(gray, 0,   255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    t_adapt    = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,11,2)
    items = {
        "Gốc":                img,
        "Tăng sáng β=+80":   brightness(img,+80),
        "Giảm sáng β=−80":   brightness(img,-80),
        "Tương phản α=2":    contrast(img,2.0),
        "Gamma γ=0.4":       gamma(img,0.4),
        "Gamma γ=2.5":       gamma(img,2.5),
        "Threshold T=127":   t_fixed,
        "Otsu Threshold":    t_otsu,
        "Adaptive Threshold":t_adapt,
    }
    fig, axes = plt.subplots(3,3,figsize=(14,12))
    fig.suptitle("Thao tác trên Điểm Ảnh",fontsize=15,fontweight="bold")
    for ax,(title,data) in zip(axes.flat,items.items()):
        ax.imshow(data if data.ndim==2 else cv2.cvtColor(data,cv2.COLOR_BGR2RGB),
                  cmap="gray" if data.ndim==2 else None)
        ax.set_title(title,fontsize=10); ax.axis("off")
    plt.tight_layout()
    os.makedirs("images",exist_ok=True)
    plt.savefig("images/pixel_operations.png",dpi=150); plt.show()

if __name__ == "__main__":
    demo(load_or_create("../samples/test.jpg"))
'

create_if_missing "02_preprocessing/geometric_transforms.py" \
'"""
geometric_transforms.py – Chủ đề 2
======================================
Biến đổi hình học: Translation, Rotation, Affine, Perspective.
Thay đổi VỊ TRÍ pixel, không thay đổi giá trị màu.

Chạy: python 02_preprocessing/geometric_transforms.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cv2, numpy as np, matplotlib.pyplot as plt
from utils.image_utils import load_or_create

def demo_transforms(img):
    h, w = img.shape[:2]; cx, cy = w//2, h//2
    M_trans = np.float32([[1,0,80],[0,1,40]])
    M_rot   = cv2.getRotationMatrix2D((cx,cy), 30, 1.0)
    src_aff = np.float32([[0,0],[w-1,0],[0,h-1]])
    dst_aff = np.float32([[50,30],[w-80,20],[30,h-60]])
    M_aff   = cv2.getAffineTransform(src_aff, dst_aff)
    src_per = np.float32([[0,0],[w-1,0],[0,h-1],[w-1,h-1]])
    dst_per = np.float32([[60,40],[w-80,20],[30,h-70],[w-50,h-40]])
    M_per   = cv2.getPerspectiveTransform(src_per, dst_per)

    results = {
        "Gốc":          cv2.cvtColor(img,cv2.COLOR_BGR2RGB),
        "Translation":  cv2.cvtColor(cv2.warpAffine(img,M_trans,(w,h)),cv2.COLOR_BGR2RGB),
        "Rotation 30°": cv2.cvtColor(cv2.warpAffine(img,M_rot,(w,h)),cv2.COLOR_BGR2RGB),
        "Flip ngang":   cv2.cvtColor(cv2.flip(img,1),cv2.COLOR_BGR2RGB),
        "Affine":       cv2.cvtColor(cv2.warpAffine(img,M_aff,(w,h)),cv2.COLOR_BGR2RGB),
        "Perspective":  cv2.cvtColor(cv2.warpPerspective(img,M_per,(w,h)),cv2.COLOR_BGR2RGB),
    }
    fig, axes = plt.subplots(2,3,figsize=(15,9))
    fig.suptitle("Biến đổi Hình Học",fontsize=14,fontweight="bold")
    for ax,(title,data) in zip(axes.flat,results.items()):
        ax.imshow(data); ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    os.makedirs("images",exist_ok=True)
    plt.savefig("images/geometric_transforms.png",dpi=150); plt.show()

if __name__ == "__main__":
    demo_transforms(load_or_create("../samples/test.jpg"))
'

create_if_missing "02_preprocessing/filtering_denoising.py" \
'"""
filtering_denoising.py – Chủ đề 2
=====================================
Lọc ảnh và khử nhiễu: Gaussian, Median, Bilateral, Sharpening.
Công thức: g(x,y) = Σ f(x+i, y+j)·h(i,j)  (convolution)

Chạy: python 02_preprocessing/filtering_denoising.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cv2, numpy as np, matplotlib.pyplot as plt
from utils.image_utils import load_or_create

def add_noise(gray, kind="gaussian"):
    out = gray.astype(np.float64)
    if kind == "gaussian":
        out += np.random.normal(0, 25, gray.shape)
    elif kind == "salt_pepper":
        out[np.random.rand(*gray.shape) < 0.04] = 255
        out[np.random.rand(*gray.shape) < 0.04] = 0
    return np.clip(out, 0, 255).astype(np.uint8)

def sharpen(img):
    blur = cv2.GaussianBlur(img,(0,0),sigmaX=3)
    return cv2.addWeighted(img,1.5,blur,-0.5,0)

def demo_filters(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ng   = add_noise(gray, "gaussian")
    nsp  = add_noise(gray, "salt_pepper")
    rows = {
        "Gốc":               (gray, cv2.GaussianBlur(gray,(5,5),0), cv2.medianBlur(gray,5)),
        "Nhiễu Gaussian":    (ng,   cv2.GaussianBlur(ng,  (5,5),0), cv2.medianBlur(ng,  5)),
        "Nhiễu Muối Tiêu":   (nsp,  cv2.GaussianBlur(nsp, (5,5),0), cv2.medianBlur(nsp, 5)),
    }
    col_labels = ["Ảnh","Gaussian Filter","Median Filter"]
    fig, axes = plt.subplots(3,3,figsize=(14,12))
    fig.suptitle("So sánh bộ lọc khử nhiễu",fontsize=14,fontweight="bold")
    for j,col in enumerate(col_labels):
        axes[0][j].set_title(col,fontsize=10,fontweight="bold")
    for i,(row_lbl,(orig,gauss,med)) in enumerate(rows.items()):
        for j,data in enumerate([orig,gauss,med]):
            axes[i][j].imshow(data,cmap="gray"); axes[i][j].axis("off")
        axes[i][0].set_ylabel(row_lbl,fontsize=10)
    plt.tight_layout()
    os.makedirs("images",exist_ok=True)
    plt.savefig("images/filtering.png",dpi=150); plt.show()
    print("Nhận xét: Median tốt hơn Gaussian khi nhiễu muối tiêu")
    print("Bilateral giữ nguyên biên trong khi Gaussian làm mờ cả biên")

if __name__ == "__main__":
    demo_filters(load_or_create("../samples/test.jpg"))
'

# ── 03_morphology ────────────────────────────────────────────
create_if_missing "03_morphology/morphological_ops.py" \
'"""
morphological_ops.py – Chủ đề 3
===================================
7 phép toán hình thái học: Erosion, Dilation, Opening,
Closing, Gradient, Top-hat, Black-hat.

Chạy: python 03_morphology/morphological_ops.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cv2, numpy as np, matplotlib.pyplot as plt
from utils.image_utils import load_or_create

def get_binary(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    _,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binary

def demo_basic_ops(binary):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    ops = {
        "Gốc (binary)":    binary,
        "Erosion ⊖":       cv2.erode(binary,se,iterations=2),
        "Dilation ⊕":      cv2.dilate(binary,se,iterations=2),
        "Opening ∘":       cv2.morphologyEx(binary,cv2.MORPH_OPEN,se),
        "Closing •":       cv2.morphologyEx(binary,cv2.MORPH_CLOSE,se),
        "Gradient (Biên)": cv2.morphologyEx(binary,cv2.MORPH_GRADIENT,se),
        "Top-hat":         cv2.morphologyEx(binary,cv2.MORPH_TOPHAT,se),
        "Black-hat":       cv2.morphologyEx(binary,cv2.MORPH_BLACKHAT,se),
    }
    fig, axes = plt.subplots(2,4,figsize=(18,9))
    fig.suptitle("7 Phép Toán Hình Thái Học",fontsize=15,fontweight="bold")
    for ax,(title,img) in zip(axes.flat,ops.items()):
        ax.imshow(img,cmap="gray"); ax.set_title(title,fontsize=11); ax.axis("off")
    plt.tight_layout()
    os.makedirs("images",exist_ok=True)
    plt.savefig("images/morphological_ops.png",dpi=150); plt.show()

if __name__ == "__main__":
    img = load_or_create("../samples/test.jpg")
    demo_basic_ops(get_binary(img))
'

create_if_missing "03_morphology/edge_extraction.py" \
'"""
edge_extraction.py – Chủ đề 3
================================
So sánh: Sobel, Laplacian, Canny, Morphological Gradient.

Pipeline Canny (quan trọng nhất):
  Gray → GaussianBlur → Sobel → Non-Max Suppression → Double Threshold → Edge linking

Chạy: python 03_morphology/edge_extraction.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cv2, numpy as np, matplotlib.pyplot as plt
from utils.image_utils import load_or_create

def sobel(gray):
    gx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    gy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    return np.clip(np.sqrt(gx**2+gy**2),0,255).astype(np.uint8)

def laplacian(gray):
    return np.clip(np.abs(cv2.Laplacian(cv2.GaussianBlur(gray,(3,3),0),cv2.CV_64F)),0,255).astype(np.uint8)

def morph_gradient(gray):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    return cv2.morphologyEx(gray,cv2.MORPH_GRADIENT,se)

def compare_edges(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    results = {
        "Ảnh Gốc":             gray,
        "Sobel":               sobel(gray),
        "Laplacian":           laplacian(gray),
        "Canny (50,150)":      cv2.Canny(gray,50,150),
        "Canny (100,200)":     cv2.Canny(gray,100,200),
        "Morph Gradient":      morph_gradient(gray),
    }
    fig,axes = plt.subplots(2,3,figsize=(15,9))
    fig.suptitle("So sánh Phương pháp Phát hiện Biên",fontsize=14,fontweight="bold")
    for ax,(title,data) in zip(axes.flat,results.items()):
        ax.imshow(data,cmap="gray"); ax.set_title(title,fontsize=11); ax.axis("off")
    plt.tight_layout()
    os.makedirs("images",exist_ok=True)
    plt.savefig("images/edge_comparison.png",dpi=150); plt.show()
    print("Canny: tốt nhất – biên mỏng, chính xác, ít nhiễu")

if __name__ == "__main__":
    img  = load_or_create("../samples/test.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    compare_edges(img)
'

# ── utils ────────────────────────────────────────────────────
create_if_missing "utils/image_utils.py" \
'"""
utils/image_utils.py
======================
Hàm tiện ích dùng chung cho toàn bộ project.
"""
import cv2, numpy as np, matplotlib.pyplot as plt, os

def load_or_create(path, size=(300,400)):
    """Đọc ảnh BGR. Nếu không có → tạo ảnh mẫu."""
    img = cv2.imread(path)
    if img is not None:
        return img
    print(f"Không tìm thấy {path}. Tạo ảnh mẫu...")
    return _create_sample(size)

def _create_sample(size=(300,400)):
    h,w = size
    img = np.ones((h,w,3),dtype=np.uint8)*210
    cv2.circle(img,(w//4,h//2),min(h,w)//5,(0,0,200),-1)
    cv2.rectangle(img,(w//2,h//4),(3*w//4,3*h//4),(0,160,0),-1)
    pts = np.array([[3*w//4,h//4],[w-20,3*h//4],[w//2+20,3*h//4]])
    cv2.fillPoly(img,[pts],(0,200,200))
    cv2.putText(img,"SAMPLE",(w//4,h-20),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,50),2)
    return img

def show_images(images, title="", cols=3, save_path=None):
    """Hiển thị dict {tên: ảnh} dạng lưới."""
    n = len(images); rows = (n+cols-1)//cols
    fig,axes = plt.subplots(rows,cols,figsize=(5*cols,4*rows),squeeze=False)
    for idx,(name,img) in enumerate(images.items()):
        ax = axes[idx//cols][idx%cols]
        if img is None: ax.axis("off"); continue
        ax.imshow(img if img.ndim==2 else cv2.cvtColor(img,cv2.COLOR_BGR2RGB),
                  cmap="gray" if img.ndim==2 else None)
        ax.set_title(name,fontsize=10); ax.axis("off")
    for idx in range(n,rows*cols):
        axes[idx//cols][idx%cols].axis("off")
    if title: fig.suptitle(title,fontsize=14,fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".",exist_ok=True)
        plt.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.show()

def print_info(img, name="Image"):
    ch = img.shape[2] if len(img.shape)==3 else 1
    print(f"[{name}] {img.shape[1]}x{img.shape[0]}px | {ch}ch | {img.dtype} | "
          f"min={img.min()} max={img.max()} | {img.nbytes/1024:.1f}KB")
'

create_if_missing "utils/__init__.py" ""

# ── requirements.txt ─────────────────────────────────────────
create_if_missing "requirements.txt" \
'opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-image>=0.21.0
scikit-learn>=1.3.0
scipy>=1.11.0
Pillow>=10.0.0
'

# ── .gitignore ───────────────────────────────────────────────
create_if_missing ".gitignore" \
'# Python
__pycache__/
*.py[cod]
*.pyo
.env
venv/
.venv/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Output ảnh (không commit ảnh output)
*/images/*.png
*/images/*.jpg
outputs/

# Ảnh test lớn (sample nhỏ thì ok commit)
samples/*.jpg
samples/*.png
samples/*.bmp

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
'

# ════════════════════════════════════════════════════════════
# BƯỚC 4 – Tạo README cho từng thư mục còn thiếu
# ════════════════════════════════════════════════════════════
echo ""
echo "[ 4/6 ] Tạo README từng thư mục..."

create_if_missing "01_color_representation/README.md" \
'# Chủ đề 1 – Biểu Diễn Ảnh · Kênh Màu · Hệ Màu

## File
| File | Nội dung |
|---|---|
| `color_spaces.py` | BGR ↔ RGB / HSV / LAB / Grayscale + lọc màu HSV |
| `histogram.py` | Histogram, Global EQ, CLAHE |

## Hệ màu quan trọng
| Hệ màu | Dùng khi |
|---|---|
| BGR/RGB | Đọc/hiển thị mặc định (OpenCV = BGR!) |
| Grayscale | Không cần màu, tiết kiệm tính toán |
| HSV | **Lọc theo màu sắc** (tách vật đỏ, xanh…) |
| LAB | So sánh màu chính xác |

## Mẹo thi
- OpenCV đọc theo **BGR** → dùng `COLOR_BGR2RGB` khi show matplotlib
- HSV tốt hơn RGB khi lọc màu vì tách H (màu) khỏi V (sáng)
- `Gray = 0.299·R + 0.587·G + 0.114·B`
'

create_if_missing "02_preprocessing/README.md" \
'# Chủ đề 2 – Tiền Xử Lý Ảnh

## File
| File | Nội dung |
|---|---|
| `pixel_operations.py` | Sáng, tương phản, gamma, threshold |
| `geometric_transforms.py` | Translation, Rotation, Affine, Perspective |
| `filtering_denoising.py` | Gaussian, Median, Bilateral, Sharpening |

## Bộ lọc nhanh
| Bộ lọc | Tốt cho | Giữ biên? |
|---|---|---|
| Gaussian | Nhiễu Gaussian | Không |
| **Median** | **Nhiễu muối tiêu** | Gần như có |
| **Bilateral** | Mọi loại | **Có** |

## Mẹo thi
- Kernel phải là số **lẻ**: 3×3, 5×5, 7×7
- Muối tiêu → **Median**; giữ biên → **Bilateral**
'

create_if_missing "03_morphology/README.md" \
'# Chủ đề 3 – Hình Thái Học

## File
| File | Nội dung |
|---|---|
| `morphological_ops.py` | 7 phép toán + so sánh SE |
| `edge_extraction.py` | Sobel, Laplacian, Canny, Morph Gradient |

## 7 phép toán
| Phép toán | = | Tác dụng |
|---|---|---|
| Erosion ⊖ | – | Thu nhỏ, loại nhiễu nhỏ |
| Dilation ⊕ | – | Phình to, nối vùng |
| **Opening** | Erosion→Dilation | **Xóa nhiễu nhỏ** |
| **Closing** | Dilation→Erosion | **Lấp lỗ nhỏ** |
| **Gradient** | Dil−Ero | **Trích biên** |
| Top-hat | Gốc−Opening | Vùng sáng nhỏ |
| Black-hat | Closing−Gốc | Vùng tối nhỏ |

## Pipeline chuẩn
`Threshold → Opening → Closing → Gradient`
'

create_if_missing "04_HOG_SVM/README.md" \
'# Chủ đề 4 – Nhận Dạng Đối Tượng: HOG + SVM

> Hướng duy nhất: **Nhận dạng/phát hiện dựa trên đặc trưng cổ điển**

## File
| File | Nội dung |
|---|---|
| `hog_features.py` | HOG lý thuyết + trực quan hoá từng bước |
| `hog_svm_classifier.py` | HOG + SVM: train/test + Sliding Window + NMS |
| `sift_orb_features.py` | SIFT/ORB keypoints, descriptor, matching |

## Pipeline HOG + SVM
```
Ảnh → Grayscale → Gradient (Gx,Gy) → Cell histogram (9 bins)
    → Block normalize (L2) → Feature vector → SVM.predict
```

## Pipeline Detection (Sliding Window)
```
Ảnh → Image Pyramid → Sliding Window → HOG → SVM score
    → Non-Maximum Suppression (NMS) → Bounding boxes
```

## So sánh HOG vs SIFT vs ORB
| | HOG | SIFT | ORB |
|---|---|---|---|
| Mục đích | Phân loại/detect | Matching | Matching real-time |
| Descriptor | Float cố định | 128-dim | 256-bit binary |
| Tốc độ | Trung bình | Chậm | **Nhanh ~10× SIFT** |
| License | Tự do | Free (4.4+) | **Tự do** |

## Mẹo thi
- HOG đi với **SVM** → giải thích được từng bước gradient → cell → block
- SIFT hỏi "tại sao bất biến scale?" → trả lời: **DoG pyramid**
- ORB dùng **Hamming distance**, SIFT dùng **L2 (Euclidean)**
'

# ════════════════════════════════════════════════════════════
# BƯỚC 5 – Tạo ảnh mẫu nhanh bằng Python
# ════════════════════════════════════════════════════════════
echo ""
echo "[ 5/6 ] Tạo ảnh mẫu (samples/test.jpg)..."

python3 - <<'PYEOF'
import cv2, numpy as np, os
os.makedirs("samples", exist_ok=True)
if not os.path.exists("samples/test.jpg"):
    h, w = 400, 600
    img = np.ones((h,w,3), dtype=np.uint8) * 210
    cv2.circle(img, (w//4, h//2), 80, (0,0,200), -1)
    cv2.rectangle(img, (w//2,h//4), (3*w//4,3*h//4), (0,160,0), -1)
    pts = np.array([[3*w//4,h//4],[w-30,3*h//4],[w//2+30,3*h//4]])
    cv2.fillPoly(img, [pts], (0,200,200))
    cv2.ellipse(img,(w//2,h//4),(80,50),45,0,360,(180,0,0),-1)
    cv2.putText(img,"SAMPLE IMAGE",(w//4,h-20),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,50),2)
    cv2.imwrite("samples/test.jpg", img)
    print("  + Đã tạo: samples/test.jpg")
else:
    print("  ~ samples/test.jpg đã có, giữ nguyên")
PYEOF

# ════════════════════════════════════════════════════════════
# BƯỚC 6 – Hiển thị cấu trúc cuối cùng
# ════════════════════════════════════════════════════════════
echo ""
echo "[ 6/6 ] Cấu trúc thư mục sau khi tái cấu trúc:"
echo ""

if command -v tree &> /dev/null; then
  tree -I "__pycache__|*.pyc|venv|.git" --dirsfirst -C
else
  find . -not -path "./.git/*" -not -path "./__pycache__/*" \
         -not -name "*.pyc" -not -path "*/venv/*" \
    | sort | sed 's|[^/]*/|  |g; s|  \([^ ]\)|── \1|'
fi

echo ""
echo "========================================"
echo -e "${GREEN}  XONG! Repo đã được tái cấu trúc.${NC}"
echo "========================================"
echo ""
echo "  Bước tiếp theo:"
echo "  1. pip install -r requirements.txt"
echo "  2. Kiểm tra: python 01_color_representation/color_spaces.py"
echo "  3. git add . && git commit -m 'refactor: restructure project layout'"
echo ""

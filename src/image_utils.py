"""
utils/image_utils.py
======================
Hàm tiện ích dùng chung cho toàn bộ project.
"""
import cv2, numpy as np, matplotlib.pyplot as plt, os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import FOOD101_SUBSET, SELECTED_CLASSES, OUTPUTS_DIR
from src.color_spaces import convert_color_spaces, normalize_image
SAVE_DIR = OUTPUTS_DIR / "01_color_representation"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
def plot_histograms(img_path: str | Path, class_name: str,
                    save: bool = True) -> None:
    img = cv2.imread(str(img_path))
    rgb, hsv, lab = convert_color_spaces(img)
    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    fig.suptitle(f"Histogram Analysis – {class_name}", fontsize=14)
    configs = [
        (rgb, ["R", "G", "B"],  ["red", "green", "blue"]),
        (hsv, ["H", "S", "V"],  ["orange", "gray", "purple"]),
        (lab, ["L", "a", "b*"], ["black", "green", "blue"]),
    ]
    for row, (space, names, colors) in enumerate(configs):
        for col, (ch, name, color) in enumerate(
                zip(cv2.split(space), names, colors)):
            axes[row][col].hist(ch.ravel(), bins=256,
                                color=color, alpha=0.7)
            axes[row][col].set_title(
                f"{['RGB', 'HSV', 'Lab'][row]} – {name}")
            axes[row][col].set_xlim(0, 255)
    plt.tight_layout()
    if save:
        out_path = SAVE_DIR / f"hist_{class_name}.png"
        plt.savefig(out_path, dpi=100)
        print(f"Saved: {out_path}")
    plt.show()

def compare_classes(classes: list = None, n_samples: int = 1) -> None:
    if classes is None:
        classes = ["pizza", "sushi"]
    for cls in classes:
        cls_dir = FOOD101_SUBSET / "train" / cls
        samples = list(cls_dir.glob("*.jpg"))[:n_samples]
        for img_path in samples:
            plot_histograms(img_path, cls)

def demo_normalize(img_path: str | Path) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return
    lab_norm = normalize_image(img)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original (BGR→RGB)")
    titles = ["L (normalized)", "a (normalized)", "b (normalized)"]
    cmaps = ["gray", "RdYlGn", "PuOr"]
    for i in range(3):
        axes[i+1].imshow(lab_norm[:, :, i], cmap=cmaps[i])
        axes[i+1].set_title(titles[i])
    for ax in axes:
        ax.axis("off")
    plt.suptitle("Lab Normalization", fontsize=13)
    plt.tight_layout()
    out_path = SAVE_DIR / "lab_normalization.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    train_dir = FOOD101_SUBSET
    print("\n Histogram Analysis")
    compare_classes(["pizza", "sushi"])
    print("\n Lab Normalization")
    sample = next((train_dir / "pizza").glob("*.jpg"), None)
    if sample:
        demo_normalize(sample)

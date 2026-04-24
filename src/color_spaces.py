import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
def convert_color_spaces(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    return rgb, hsv, lab

def plot_histograms(img_path, class_name):
    img = cv2.imread(str(img_path))
    rgb, hsv, lab = convert_color_spaces(img)
    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    fig.suptitle(f"Histogram Analysis – {class_name}", fontsize=14)
    configs = [
        (rgb, ["R","G","B"],   ["red","green","blue"]),
        (hsv, ["H","S","V"],   ["orange","gray","purple"]),
        (lab, ["L","a","b*"],  ["black","green","blue"]),
    ]
    for row, (space, names, colors) in enumerate(configs):
        for col, (ch, name, color) in enumerate(zip(
                cv2.split(space), names, colors)):
            axes[row][col].hist(ch.ravel(), bins=256, color=color, alpha=0.7)
            axes[row][col].set_title(f"{['RGB','HSV','Lab'][row]} – {name}")
            axes[row][col].set_xlim(0, 255)
    plt.tight_layout()
    plt.savefig(f"hist_{class_name}.png", dpi=100)
    plt.show()

def normalize_image(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab[:,:,0] /= 100.0 
    lab[:,:,1] = (lab[:,:,1] + 128) / 255.0 
    lab[:,:,2] = (lab[:,:,2] + 128) / 255.0  
    return lab
dataset_dir = Path(r"C:\Users\ADMIN\Downloads\food101_subset\train")
sample_pizza  = next((dataset_dir / "pizza").glob("*.jpg"))
sample_sushi  = next((dataset_dir / "sushi").glob("*.jpg"))
plot_histograms(sample_pizza, "pizza")
plot_histograms(sample_sushi, "sushi")

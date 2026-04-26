import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.filtering import show_preprocess_steps
from src.config import FOOD101_SUBSET
import cv2
if __name__ == "__main__":
    sample = next((FOOD101_SUBSET / "train" / "pizza").glob("*.jpg"), None)
    if sample:
        img = cv2.imread(str(sample))
        save_path = Path(__file__).parent / "images" / "preprocessing_steps.png"
        show_preprocess_steps(img, save_path=save_path)
        print(f"Saved sample to: {save_path}")
    else:
        print("No sample image found")

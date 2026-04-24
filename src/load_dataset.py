import os, json, shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
ROOT = Path(r"C:\Users\ADMIN\Downloads\archive(1)\food-101\food-101")
SELECTED_CLASSES = [
    "pizza", "hamburger", "french_fries", "ice_cream", "chocolate_cake",
    "sushi", "ramen", "fried_rice", "omelette", "pancakes",
    "hot_dog", "grilled_salmon", "caesar_salad", "donuts", "dumplings",
]
with open(ROOT / "meta/train.json") as f:
    train_json = json.load(f)
with open(ROOT / "meta/test.json") as f:
    test_json = json.load(f)
OUT = Path(r"C:\Users\ADMIN\Downloads\food101_subset")

def copy_split(data_dict, split_name):
    for cls in SELECTED_CLASSES:
        paths = data_dict.get(cls, [])
        if split_name == "train":
            train_p, val_p = train_test_split(paths, test_size=0.2, random_state=42)
            _copy(train_p, "train", cls)
            _copy(val_p,   "val",   cls)
        else:
            _copy(paths, "test", cls)

def _copy(paths, split, cls):
    dst = OUT / split / cls
    dst.mkdir(parents=True, exist_ok=True)
    for p in paths:
        src = ROOT / "images" / (p + ".jpg")
        shutil.copy(src, dst / (Path(p).name + ".jpg"))

copy_split(train_json, "train")
copy_split(test_json,  "test")
for split in ["train", "val", "test"]:
    total = sum(len(list((OUT/split/c).glob("*.jpg"))) for c in SELECTED_CLASSES)
    print(f"{split}: {total} ảnh")

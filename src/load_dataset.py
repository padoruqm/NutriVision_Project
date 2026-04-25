import json, shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import FOOD101_RAW, FOOD101_SUBSET, SELECTED_CLASSES

def copy_split(data_dict, split_name, out_dir):
    for cls in SELECTED_CLASSES:
        paths = data_dict.get(cls, [])
        if split_name == "train":
            tr, va = train_test_split(paths, test_size=0.2, random_state=42)
            _copy(tr, "train", cls, out_dir)
            _copy(va, "val",   cls, out_dir)
        else:
            _copy(paths, "test", cls, out_dir)

def _copy(paths, split, cls, out_dir):
    dst = out_dir / split / cls
    dst.mkdir(parents=True, exist_ok=True)
    for p in paths:
        src = FOOD101_RAW / "images" / (p + ".jpg")
        if src.exists():
            shutil.copy(src, dst / (Path(p).name + ".jpg"))

def load_and_split():
    meta = FOOD101_RAW / "meta"
    if not meta.exists():
        print(f"Food-101 not found at {FOOD101_RAW}")
        return
    with open(meta / "train.json") as f:
        train_json = json.load(f)
    with open(meta / "test.json") as f:
        test_json = json.load(f)
    copy_split(train_json, "train", FOOD101_SUBSET)
    copy_split(test_json,  "test",  FOOD101_SUBSET)
    for s in ["train", "val", "test"]:
        d = FOOD101_SUBSET / s
        if d.exists():
            n = sum(len(list((d/c).glob("*.jpg"))) for c in SELECTED_CLASSES)
            print(f"  {s}: {n}")

if __name__ == "__main__":
    load_and_split()

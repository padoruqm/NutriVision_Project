from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR          = PROJECT_ROOT / "data"
FOOD101_RAW       = DATA_DIR / "food-101"
FOOD101_SUBSET    = DATA_DIR / "food101_subset"
FOOD101_PROCESSED = DATA_DIR / "food101_processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SELECTED_CLASSES = [
    "pizza", "hamburger", "french_fries", "ice_cream", "chocolate_cake",
    "sushi", "ramen", "fried_rice", "omelette", "pancakes",
    "hot_dog", "grilled_salmon", "caesar_salad", "donuts", "dumplings",
]
TARGET_SIZE = (224, 224)

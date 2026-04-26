### Prerequisites
- Python 3.8+
- Food-101 dataset downloaded to `data/food-101/`
### Run Preprocessing
To preprocess the entire dataset in one command:
```bash
python run_preprocessing.py
```
This will:
1. Load and split Food-101 dataset into train/val/test
2. Apply preprocessing (resize, bilateral filter, CLAHE)
3. Save processed images to `data/food101_processed/`
### Selected Food Classes
The project uses 15 food classes:
- pizza, hamburger, french_fries, ice_cream, chocolate_cake
- sushi, ramen, fried_rice, omelette, pancakes
- hot_dog, grilled_salmon, caesar_salad, donuts, dumplings

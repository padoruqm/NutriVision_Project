## Cài đặt
```
pip install opencv-python numpy matplotlib scikit-learn scikit-image tqdm
```
Giải nén vào `data/food-101/` (phải có `images/` và `meta/` bên trong).
## Chạy
```bash
cd NutriVision_Project
# buoc 1: chia dataset
python -m src.load_dataset
# buoc 2: tien xu ly -> .npy
python -m src.filtering
```
Kết quả nằm trong `data/food101_subset/` và `data/food101_processed/`.

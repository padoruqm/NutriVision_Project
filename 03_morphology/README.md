## Pipeline Tổng Thể

1. **Preprocessing**  
   Denoise (Bilateral Filter) → CLAHE (tăng tương phản) → Resize + Reflect Padding → 224×224  
   → Lưu vào `data/food101_processed`

2. **Morphology**  
   Sử dụng `combine_lab_otsu` để tạo contour chất lượng cao.

3. **Segmentation (GrabCut Hybrid)** ← **Bước chính hiện tại**  
   Sử dụng contour từ Morphology làm seed cho GrabCut → cho mask liền mạch và chính xác cao.  
   → Lưu kết quả vào `data/food101_segmentation`

### Đầu ra của Segmentation

- crop_xxxx.jpg: Tight crop của vùng thức ăn chính (đã resize giữ tỷ lệ + padding). Đây là ảnh sẵn sàng đưa vào HOG / feature extraction / classification.

- mask_xxxx.png: Binary mask của vùng thức ăn. Dùng để: Tính diện tích thức ăn, Visualize

## Cách chạy Pipeline

### Bước 1: Tiền xử lý ảnh

```bash
cd 02_preprocessing
python run_preprocessing.py
```

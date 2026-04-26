Thư mục này chứa mã nguồn xử lý ảnh từ dữ liệu thô (Raw Data) sang dữ liệu sạch (Processed Data) sẵn sàng cho việc huấn luyện mô hình AI.

## 1. Quy trình chuẩn bị dữ liệu

Trước khi chạy script, bạn cần đảm bảo dữ liệu được đặt đúng cấu trúc thư mục sau:

1.  **Tải Dataset:** Tải bộ dữ liệu [Food-101 từ Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101).
2.  **Cấu trúc thư mục `data/`:**
    ```text
    NutriVision_Project/
    └── data/
        └── food-101/           <-- Giải nén dataset vào đây
            ├── images/
            └── meta/
    ```
3.  **Khởi tạo Subset:** Chạy file `load_dataset.py` để trích xuất 15 lớp quan trọng (theo `config.py`) nhằm giảm nhẹ dung lượng tính toán.
    ```bash
    python load_dataset.py
    ```
    _Kết quả: Tạo ra thư mục `data/food101_subset/`._

---

## 2. Hướng dẫn chạy Preprocessing

Để tiến hành tiền xử lý toàn bộ tập subset (khử nhiễu, cân bằng sáng, resize chuẩn hóa), bạn hãy chạy lệnh sau từ thư mục gốc của Project:

```bash
python 02_preprocessing/script.py
```

## 3. Kỹ thuật Resize giữ nguyên tỷ lệ (Aspect Ratio)

Khác với cách resize "cứng" (ép ảnh về hình vuông gây méo hình), Pipeline này sử dụng phương pháp **`resize_keep_ratio`** trong class `ImageEnhancer`:

### Cơ chế hoạt động:

1.  **Tính toán tỷ lệ:** Tìm cạnh dài nhất của ảnh và tính toán tỷ lệ thu nhỏ sao cho cạnh đó bằng `224px`.
2.  **Resize bảo toàn:** Thu nhỏ ảnh theo tỷ lệ vừa tính. Lúc này ảnh vẫn giữ đúng hình dạng gốc nhưng có một cạnh chưa đủ `224px`.
3.  **Padding (Đệm ảnh):** Sử dụng hàm `cv2.copyMakeBorder` để thêm các dải màu đen vào các cạnh còn thiếu, đưa ảnh về kích thước chuẩn `224x224`.

**Ưu điểm:**

- Không làm biến dạng đặc trưng hình học của món ăn (ví dụ: pizza vẫn hình tròn thay vì hình bầu dục).
- Giúp mô hình học được hình dạng thực tế của vật thể tốt hơn.

---

## 4. Cách sử dụng dữ liệu sạch ở các bước sau

Dữ liệu sau khi chạy `script.py` đã nằm gọn gàng trong `data/02_preprocessing/`. Để gọi dữ liệu này trong các bước tiếp theo, bạn nên sử dụng biến cấu hình trong `src/config.py`.

### Ví dụ cách gọi:

```python
from src.config import FOOD101_PROCESSED
import cv2

# Lấy danh sách các món ăn
classes = [f.name for f in FOOD101_PROCESSED.iterdir() if f.is_dir()]

# Đọc một ảnh bất kỳ để xử lý bước tiếp theo
sample_img_path = next(FOOD101_PROCESSED.rglob("*.jpg"))
img = cv2.imread(str(sample_img_path))
```

---

## 5. Các bước tiền xử lý chi tiết (Pipeline)

Mỗi bức ảnh sẽ đi qua các bộ lọc sau:

1.  **Resize & Padding:** Chuẩn hóa về 224x224 (không méo).
2.  **Bilateral Filter:** Khử nhiễu nhưng vẫn giữ sắc nét các đường biên (edge-preserving).
3.  **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Tự động cân bằng độ sáng và tăng độ tương phản cục bộ, giúp làm nổi bật các chi tiết bề mặt thực phẩm.

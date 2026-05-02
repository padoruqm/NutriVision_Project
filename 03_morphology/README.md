Thư mục này đảm nhiệm vai trò **Contour Dêtction**. Sau khi ảnh đã được làm sạch và tăng cường ở Bước 02, chúng ta sẽ áp dụng các kỹ thuật hình thái học (Morphology) và phát hiện biên (Edge Detection) để tìm ra đường viền (contour) bao quanh món ăn.

## 🎯 1. Chức năng cốt lõi

- **Phát hiện biên (Edge Detection):** Sử dụng thuật toán `Canny` kết hợp tính toán ngưỡng tự động (Auto Canny dựa trên Median) để tìm ra các đường nét chính của vật thể.
- **Hình thái học (Morphology):** Sử dụng phép toán `Morphology Close` (Giãn nở kết hợp Xói mòn) để lấp đầy các khoảng trống, đứt gãy trên đường biên.
- **Lọc Contour:** Giả định thức ăn có diện tích lớn nhất trong ảnh nên tìm và chọn ra dải viền (contour) có diện tích lớn nhất và hợp lý nhất (loại bỏ nhiễu nhỏ hoặc viền bao toàn bộ ảnh) để đại diện cho món ăn.
- **Chiến lược được chọn:** Sau khi thử nghiệm 4 chiến lược (`gray_canny`, `otsu_sv`, `lab_canny`, `gradient_canny`), dự án quyết định sử dụng **`gray_canny`** vì nó có vẻ ổn định nhất trên nhiều ảnh, còn 3 chiến lược kia đôi khi bị thiếu hoặc thừa biên.

---

## 2. Hướng dẫn chạy code

**Lưu ý:** Bạn PHẢI chạy Step 02 (Preprocessing) trước để có dữ liệu tại thư mục `data/food101_processed`.

### Lựa chọn 1: Chạy chế độ kiểm thử (Testing/Visualization)

Nếu bạn muốn quan sát và so sánh trực quan hiệu quả của 4 phương pháp trên 50 ảnh ngẫu nhiên, hãy chạy:

```bash
python 03_morphology/morphology.py
```

### Lựa chọn 2: Chạy quy trình hàng loạt (Pipeline)

Để chính thức vẽ contour cho toàn bộ dữ liệu dự án và lưu lại, hãy chạy:

```bash
python 03_morphology/script.py
```

## 3. Cách gọi dữ liệu đã Contour cho các bước tiếp theo

```bash
import cv2
import matplotlib.pyplot as plt
from src.config import FOOD101_CONTOUR

# 1. Lấy danh sách các thư mục món ăn
classes = [d.name for d in FOOD101_CONTOUR.iterdir() if d.is_dir()]
print(f"Các lớp món ăn: {classes}")

# 2. Đọc thử một ảnh ngẫu nhiên từ lớp 'pizza'
pizza_dir = FOOD101_CONTOUR / "pizza"
sample_img_path = next(pizza_dir.glob("*.jpg"))

# 3. Đọc ảnh bằng OpenCV (Ảnh này đã được vẽ viền xanh lá)
img_bgr = cv2.imread(str(sample_img_path))

# Hiển thị ảnh
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title(f"Contoured Image: {sample_img_path.name}")
plt.axis('off')
plt.show()
```

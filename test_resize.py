import cv2
import matplotlib.pyplot as plt

path = "/Users/quangminh/Documents/University/Computer vision/Project/Data/food101_full_15classes/ice_cream/15420.jpg"

def load_or_create(path, size=(300,400)):
    """Đọc ảnh BGR. Nếu không có → tạo ảnh mẫu."""
    img = cv2.imread(path)
    if img is not None:
        return img

import matplotlib.pyplot as plt

img = load_or_create(path)

resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img_rgb.shape)

plt.imshow(img_rgb)
plt.title("Image")
plt.axis("off")
plt.show()
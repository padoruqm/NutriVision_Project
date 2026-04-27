# Chủ đề 4 – Nhận Dạng Đối Tượng: HOG + SVM

> Hướng duy nhất: **Nhận dạng/phát hiện dựa trên đặc trưng cổ điển**

## File
| File | Nội dung |
|---|---|
| `hog_features.py` | HOG lý thuyết + trực quan hoá từng bước |
| `sift_orb_features.py` | SIFT/ORB keypoints, descriptor, matching |

## Pipeline HOG + SVM
```
Ảnh → Grayscale → Gradient (Gx,Gy) → Cell histogram (9 bins)
    → Block normalize (L2) → Feature vector → SVM.predict
```

## Pipeline Detection (Sliding Window)
```
Ảnh → Image Pyramid → Sliding Window → HOG → SVM score
    → Non-Maximum Suppression (NMS) → Bounding boxes
```

## So sánh HOG vs SIFT vs ORB
| | HOG | SIFT | ORB |
|---|---|---|---|
| Mục đích | Phân loại/detect | Matching | Matching real-time |
| Descriptor | Float cố định | 128-dim | 256-bit binary |
| Tốc độ | Trung bình | Chậm | **Nhanh ~10× SIFT** |
| License | Tự do | Free (4.4+) | **Tự do** |

## Mẹo thi
- HOG đi với **SVM** → giải thích được từng bước gradient → cell → block
- SIFT hỏi "tại sao bất biến scale?" → trả lời: **DoG pyramid**
- ORB dùng **Hamming distance**, SIFT dùng **L2 (Euclidean)**


## Chay toan bo Pipeline
# - Chạy toàn bộ pipeline (train + đánh giá)
python3 04_HOG_SVM/run_pipeline.py --dataset dataset/
# - Chạy + dự đoán luôn 1 ảnh test
python3 04_HOG_SVM/run_pipeline.py --dataset dataset/ --test samples/test.jpg


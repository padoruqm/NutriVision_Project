# Chủ đề 4 – Nhận Dạng Đối Tượng: HOG + SVM

## File
| File | Nội dung |
| `hog_features.py` | HOG lý thuyết + trực quan hoá từng bước |
| `run_pipeline.py` | Chạy pipeline HOG + SVM |
| `run_pipeline_2.py` | Chạy pipeline HOG + Color Histogram + SVM|

## Chay toan bo Pipeline
# - Chạy toàn bộ pipeline (train + đánh giá)
python3 04_HOG_SVM/run_pipeline.py --dataset dataset/
# - Chạy + dự đoán luôn 1 ảnh test
python3 04_HOG_SVM/run_pipeline.py --dataset dataset/ --test samples/test.jpg

# - Chạy toàn bộ pipeline (train + đánh giá)
python3 04_HOG_SVM/run_pipeline_2.py --dataset dataset/
# - Chạy + dự đoán luôn 1 ảnh test
python3 04_HOG_SVM/run_pipeline_2.py --dataset dataset/ --test samples/test.jpg



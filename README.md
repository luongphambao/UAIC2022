

# UAIC2022-DeadlineKLTN
Nhóm sử dụng mô hình ABCnet cho giai đoạn detection và VietOCR+vedastr cho giai đoạn recognition (mô hình phân loại arttext do sử dụng không hiệu quả nên đã bị bỏ đi)

## 1. Setup môi trường:
Đầu tiên ta cần cài đặt torch với cuda11.1
```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Cài  đặt detectron2
```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
```

Install các thư viện cần thiết khác
```
pip install -r requirements.txt
```
#Cài đặt text detection
```
cd ABCnet
rm -rf build
python3 setup.py build develop
```

## 2. Chuẩn bị 
Trước khi tiến hành Inference, vui lòng để toàn bộ hình ảnh cần được đánh giá vào thư mục `data/public_test_images` (chúng tôi đã có down sẵn tập test A ở file `download.sh`).
Trước khi tiến hành đánh giá vui lòng để dữ liệu hình ảnh vào thư mực `datasets/` (ví dụ nếu là tập private test thì đường dẫn tới ảnh của private test sẽ là `datasets/uaic2022_private_test/images/ `)
Để tải mô hình đã được huấn luyện ta chỉ cần chạy lệnh sau
```
cd weights
bash down_weights.sh
```
## 3. Chạy Inference



Sau đó tiến hành chạy inference bằng cách chỉnh sửa đường ảnh trong file `run.sh` (thay `data_dir` ứng với dữ liệu cần chạy )

```
bash run.sh
```







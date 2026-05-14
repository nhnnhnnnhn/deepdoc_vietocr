<p align="center">
  <a href="./README.md">Tiếng Việt</a> |
  <a href="./README_en.md">English</a> |
</p>

# *Deep*Doc + VietOCR - Công cụ OCR cho tiếng Việt nhanh và tiết kiệm chi phí

- [1. Giới thiệu](#1)
- [2. Kiến trúc kỹ thuật](#2)
- [3. Cài đặt và chạy thử](#3)
- [4. Docker (GPU Web UI)](#4)

<a name="1"></a>

## 1. Giới thiệu

Với một loạt tài liệu từ nhiều nguồn khác nhau với nhiều định dạng khác nhau và cùng với các yêu cầu truy xuất đa dạng,  một công cụ trích xuất chính xác là rất cần thiết với bất kỳ doanh nghiệp nào. Hôm nay mình sẽ giới thiệu với các bạn công cụ DeepDoc, một công cụ OCR rất nhanh và tiết kiệm chi phí khi chỉ cần chạy trên CPU. Không những vậy còn có các tính năng kèm theo là Layout Recognizer (nhận diện bố cục) và Table Structure Recognizer (nhận diện cấu trúc bảng) giúp giữ định dạng văn bản sau OCR. 

Tuy nhiên DeepDoc chưa được chuẩn hóa cho tiếng Việt nên mình đã thay VietOCR và bản ONNX vào phần Text Recognizer để có thể nhận dạng văn bảng tiếng Việt tốt hơn. Bạn cũng có thể tham khảo DeepDoc phiên bản gốc tại [đây](https://github.com/infiniflow/ragflow/blob/main/deepdoc/README.md). Thêm vào đó, DeepDoc bản chất là 1 phần xử lý dữ liệu cho luồng RAG thuộc dự án RAGFlow nên việc mình tách ra thành 1 git riêng cũng để ứng dụng có thể tùy chỉnh một cách thuận tiện hơn

<a name="2"></a>

## 2. Kiến trúc kỹ thuật
### 2.1 OCR
Phần này DeepDoc sử dụng PaddleOCR - Công cụ mã nguồn mở rất thông dụng được phát triển bởi Baidu - sau khi chuyển sang ONNX. Cơ bản thì ONNX (Open Neural Network Exchange) dạng mở cho mô hình AI, cho phép xuất – nhập mô hình giữa nhiều framework (PyTorch, TensorFlow, v.v.), giúp mô hình tương thích đa nền tảng, tối ưu tốc độ trên CPU/GPU và giảm chi phí hạ tầng khi triển khai (chúng ta sẽ không đi quá sâu về chủ đề này)

Bên DeepDoc họ không ghi rõ là sử dụng version bao nhiêu vì sau khi chuyển sang ONNX cũng khó để xác định lại. Để hiểu qua về cách hoạt động, mình sẽ tham khảo kiến trúc OCR PP-OCRv5 của bản mới nhất là PaddleOCR 3.0, bao gồm 4 phần chính:

- Image Preprocessing Module (Tiền xử lý ảnh): Cải thiện chất lượng ảnh, xử lý xoay/nghiêng bằng mô hình phân loại hướng (PP-LCNet) và unwarping (UVDoc).

- Text Detection (Phát hiện văn bản): Nâng cấp từ PP-OCRv4 nhờ backbone PP-HGNetV2, distillation từ - GOT-OCR2.0, và tăng cường dữ liệu (sinh tổng hợp, xoay, mờ, biến dạng). Giữ lại PFHead và DSR từ phiên bản trước.

- Text Line Orientation Classification (Phân loại hướng dòng chữ): Tự động phát hiện, sửa hướng dòng chữ (ngược, xoay) để chuẩn bị cho bước nhận dạng.

- Text Recognition (Nhận dạng văn bản): Kiến trúc 2 nhánh với PP-HGNetV2, huấn luyện bằng GTC-NRTR (attention) để hướng dẫn SVTR-HGNet (CTC, nhẹ, nhanh). Dữ liệu huấn luyện được tăng cường từ tài liệu, PDF, e-book, và sinh mẫu chữ viết tay.

<div align="center" style="margin-top:20px;margin-bottom:20px;">
    <img src="img\x6.png" width="900"/>
</div>

Chi tiết về PP-OCRv5, bạn có thể tham khảo tài liệu chính thức tại [đây](https://arxiv.org/html/2507.05595v1).

Như đã nói bên trên, phần Recognition của Paddle đã được thay bằng VietOCR và bản ONNX để việc nhận dạng chữ tiếng Việt chính xác hơn. Về VietOCR thì đó là 1 công cụ quá phổ biến cho OCR ở Việt Nam rồi, nên mình sẽ không đi sâu, các bạn có thể tìm hiểu thêm ở [đây](https://github.com/pbcquoc/vietocr). Còn phần chuyển sang định dạng ONNX cho VietOCR thì mình tham khảo từ bài viết [này.](https://viblo.asia/p/chuyen-doi-mo-hinh-hoc-sau-ve-onnx-bWrZnz4vZxw)

### 2.2 Layout Recognizer và Table Structure Recognizer
Phần này thì DeepDoc sử dụng YOLOv10 (You Only Look Once) - cũng là 1 phương pháp object detection (phát hiện đối tượng) phổ biến - phiên bản ONNX.

Kiến trúc cơ bản gồm 3 phần chính:

- Backbone: trích xuất đặc trưng từ ảnh, dùng thiết kế nhẹ và hiệu quả (giữ lại ý tưởng từ YOLOv8 nhưng cải tiến block để giảm tính toán).

- Neck: kết hợp đa cấp độ đặc trưng (FPN/PAN cải tiến) để phát hiện tốt cả vật thể nhỏ lẫn lớn.

- Head: sử dụng Anchor-Free decoupled head (tách nhánh classification và regression), tăng độ chính xác và dễ huấn luyện.

<div align="center" style="margin-top:20px;margin-bottom:20px;">
    <img src="img\af645ed9-7301-4ec4-81e7-cb996ddf2d7f.webp" width="900"/>
</div>


Trong DeepDoc, YOLOv10 được huấn luyện để nhận dạng các loại nhãn cho phần Layout Recognizer và Table Structure Recognizer cơ bản bao phủ hầu hết các trường hợp.

Đối với Layout Recognizer, có 10 loại:
- Text (Văn bản)
- Title (Tiêu đề)
- Image (Hình ảnh)
- Image Caption (Chú thích hình ảnh)
- Table (Bảng)
- Table Caption (Chú thích bảng)
- Header (Đầu đề)
- Footer (Chân trang)
- Reference (Tài liệu tham khảo)
- Equation (Phương trình)

Đối với Table Structure Recognizer, có 5 loại:
- Column (Cột)
- Row (Hàng)
- Column header (Đầu đề cột)
- Projected row header (Đầu đề hàng được chiếu)
- Spanning cell (Ô trải dài)


Để có thể hiểu rõ hơn về YOLOv10, bạn có thể tham khảo tài liệu chính thức tại [đây](https://arxiv.org/pdf/2405.14458).

<a name="3"></a>

## 3. Cài đặt và chạy thử

Clone repo về máy:
```bash
git clone https://github.com/nhnnhnnnhn/deepdoc_vietocr.git
cd deepdoc_vietocr
```

Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

Đặt file trọng số VietOCR vào đúng đường dẫn:
```
vietocr/weight/vgg_seq2seq.pth
```

Các file ONNX (`det.onnx`, `layout.onnx`, `tsr.onnx`) sẽ tự tải về từ HuggingFace lần đầu chạy. Nếu gặp lỗi kết nối (thường trên Windows), đặt biến môi trường trước:
```bash
set HF_ENDPOINT=https://hf-mirror.com
```

### 3.1. OCR
```bash
python t_ocr.py --inputs=path_to_images_or_pdfs --output_dir=path_to_store_result
```
Đầu vào có thể là thư mục chứa hình ảnh hoặc PDF, hoặc một file đơn lẻ. Đầu ra gồm 1 ảnh với các bounding box được nhận diện và 1 file `.txt` chứa văn bản OCR.

<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="img\Screenshot 2025-08-28 171633.png" width="900"/>
</div>

Mặc định dùng VietOCR Seq2seq — nhanh và chính xác. Có thể đổi sang Transformer trong `module/ocr.py` nhưng không khuyến nghị (chậm hơn nhiều, độ chính xác tăng không đáng kể). Nếu cần tốc độ tối đa, import `ocr_onnx` thay cho `ocr` (độ chính xác giảm nhẹ).

### 3.2. Layout Recognizer
```bash
python t_recognizer.py --inputs=path_to_images_or_pdfs --threshold=0.2 --mode=layout --output_dir=path_to_store_result
```
Đầu ra gồm 1 ảnh với nhãn bố cục:
<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="img\49806-Article Text-153529-1-10-20200804_page-0002.jpg" width="1000"/>
</div>

### 3.3. Table Structure Recognizer
```bash
python t_recognizer.py --inputs=path_to_images_or_pdfs --threshold=0.2 --mode=tsr --output_dir=path_to_store_result
```
Đầu ra gồm 1 ảnh gán nhãn và 1 file `.md` với nội dung bảng được trích xuất:
<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="img\Screenshot 2025-08-28 182132.png" width="1000"/>
</div>

### 3.4. Full Pipeline
Chạy toàn bộ pipeline: Layout → Table Markdown + OCR các vùng còn lại → file `.md` hợp nhất:
```bash
python full_pipeline.py --inputs=path_to_images_or_pdfs --output_dir=./output --threshold=0.5
```

> **Lưu ý:** Tất cả script chuyển output sang file log trong `log/`. Không có output trên console — kiểm tra `log/t_ocr.log`, `log/t_recognizer.log`, hoặc `log/full_pipeline.log`.

<a name="4"></a>

## 4. Docker (GPU Web UI)

Dự án có sẵn Docker image với giao diện web để upload file và xem kết quả OCR trực tiếp trên trình duyệt.

### 4.1. NVIDIA CUDA

**Yêu cầu:** NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Kiểm tra GPU:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

Build và chạy:
```bash
cp .env.docker.example .env
# Chỉnh sửa .env nếu cần
docker compose build
docker compose up
```

### 4.2. AMD ROCm 7

**Yêu cầu:** ROCm 7.x driver (`amdgpu-dkms`) trên host. Không cần cài ROCm user-space — đã có trong Docker image.

Kiểm tra GPU:
```bash
ls /dev/kfd /dev/dri/renderD*
```

Thêm user vào group (nếu chưa có):
```bash
sudo usermod -aG video,render $USER  # cần re-login sau lệnh này
```

Build và chạy:
```bash
cp .env.rocm.example .env
docker compose -f docker-compose.rocm.yml build
docker compose -f docker-compose.rocm.yml up
```

> **Lần đầu chạy:** MIGraphX sẽ compile các ONNX model, request đầu tiên có thể chậm hơn 30–90 giây. Các lần sau chạy bình thường.

### 4.3. Truy cập Web UI

Sau khi container khởi động:
```
http://127.0.0.1:8000
```

Xem thêm chi tiết cấu hình, biến môi trường và troubleshooting tại [DOCKER.md](./DOCKER.md).

---

## Kết
Hy vọng các bạn thấy công cụ hữu ích và áp dụng được vào thực tế. Nếu có góp ý hãy mở Issue hoặc Pull Request. Cảm ơn các bạn đã đọc!

## Tài liệu tham khảo
- DeepDoc: https://github.com/infiniflow/ragflow/blob/main/deepdoc/README.md
- PP-OCRv5: https://arxiv.org/html/2507.05595v1
- VietOCR: https://github.com/pbcquoc/vietocr
- VietOCR ONNX: https://viblo.asia/p/chuyen-doi-mo-hinh-hoc-sau-ve-onnx-bWrZnz4vZxw
- YOLOv10: https://arxiv.org/pdf/2405.14458

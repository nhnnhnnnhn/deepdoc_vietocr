# Xây dựng Pipeline Trích Xuất Tài Liệu Tiếng Việt Kết Hợp OCR và Hiệu Chuẩn LLM Phục Vụ Hệ Thống RAG

**Tác giả:** nhnnhnnnhn  
**Ngày:** Tháng 5, 2026  
**Repository:** https://github.com/nhnnhnnnhn/deepdoc_vietocr

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Cơ sở lý thuyết](#2-cơ-sở-lý-thuyết)
3. [Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
4. [Triển khai](#4-triển-khai)
5. [Đánh giá và thảo luận](#5-đánh-giá-và-thảo-luận)
6. [Kết luận và hướng phát triển](#6-kết-luận-và-hướng-phát-triển)
7. [Tài liệu tham khảo](#7-tài-liệu-tham-khảo)

---

## 1. Giới thiệu

### 1.1 Bối cảnh và động lực

Trong bối cảnh chuyển đổi số đang diễn ra mạnh mẽ tại Việt Nam, nhu cầu xử lý và khai thác thông tin từ các tài liệu giấy, PDF scan hay tài liệu hành chính ngày càng trở nên cấp thiết. Các hệ thống Retrieval-Augmented Generation (RAG) — kết hợp truy xuất tài liệu với mô hình ngôn ngữ lớn (LLM) — đang nổi lên như một giải pháp hiệu quả để xây dựng chatbot trả lời câu hỏi dựa trên nguồn dữ liệu nội bộ. Tuy nhiên, một điểm nghẽn quan trọng của bất kỳ hệ thống RAG nào chính là chất lượng của dữ liệu đầu vào: tài liệu phải được trích xuất thành văn bản có cấu trúc, đúng chính tả và đúng định dạng thì LLM mới có thể suy luận chính xác.

Tiếng Việt đặt ra những thách thức đặc thù cho hệ thống OCR so với các ngôn ngữ Latin thông thường, bao gồm hệ thống dấu thanh và dấu phụ phong phú (6 thanh điệu, nhiều ký tự ghép), sự đa dạng về font chữ và bố cục tài liệu hành chính, cũng như tình trạng chất lượng scan thấp phổ biến trong các tài liệu thực tế.

DeepDoc — một công cụ OCR mã nguồn mở được tách ra từ dự án RAGFlow của InfiniFlow — cung cấp nền tảng vững chắc với khả năng nhận diện bố cục và cấu trúc bảng. Tuy nhiên, phần nhận dạng ký tự của DeepDoc được tối ưu cho tiếng Trung và tiếng Anh, chưa xử lý tốt dấu tiếng Việt.

### 1.2 Vấn đề

OCR raw trên tài liệu tiếng Việt thường gặp các lỗi sau:

- **Lỗi dấu thanh và dấu phụ:** nhầm lẫn giữa các ký tự gần nhau (ă/a, â/a, ơ/o, ư/u) và các dấu thanh (hỏi/ngã, sắc/nặng).
- **Nhầm ký tự hình dạng tương tự:** O/0, l/I/1, S/5 — đặc biệt nguy hiểm trong các trường số liệu, ngày tháng, mã số.
- **Sai cấu trúc bảng:** các ô bị gộp hoặc tách sai do giới hạn đường kẻ không rõ.
- **Sai thứ tự đọc:** tài liệu nhiều cột bị đọc tuần tự theo chiều dọc thay vì theo cột.
- **Thiếu cấu trúc Markdown:** văn bản thu được là văn bản thuần không có tiêu đề, gây khó khăn cho chunking trong RAG.

Những lỗi này, nếu không được xử lý, sẽ làm giảm nghiêm trọng chất lượng truy xuất và câu trả lời của hệ thống RAG.

### 1.3 Đóng góp

Bài báo cáo này trình bày một pipeline hoàn chỉnh gồm các đóng góp sau:

1. **Thay thế Text Recognizer** của DeepDoc từ PaddleOCR sang VietOCR (vgg_seq2seq), cải thiện độ chính xác nhận dạng dấu tiếng Việt, đồng thời hỗ trợ cả backend PyTorch và ONNX.
2. **Hệ thống đánh giá độ tin cậy** (Confidence Scoring) tự động trên cả OCR lẫn bố cục, cung cấp số liệu khách quan về chất lượng từng trang.
3. **Phân tích rủi ro tự động** (VLM Risk Analysis) trước khi gọi LLM, xác định vùng cần ưu tiên kiểm tra.
4. **Nhận diện cấu trúc văn bản pháp lý tiếng Việt** (Heading Detection) theo quy ước tài liệu hành chính Việt Nam.
5. **Tầng hậu kiểm LLM** (AI Post-Check) sử dụng Qwen3-4B chạy local qua giao thức OpenAI-compatible API, với cơ chế kích hoạt thích nghi dựa trên độ tin cậy.
6. **Correction Memory** — hệ thống học và áp dụng lại các sửa lỗi đã xác nhận.
7. **Giao diện web** (FastAPI) và **Docker** hỗ trợ NVIDIA CUDA và AMD ROCm, phục vụ triển khai thực tế.

---

## 2. Cơ sở lý thuyết

### 2.1 Thách thức OCR tiếng Việt

Tiếng Việt sử dụng bảng chữ cái Latin mở rộng với 29 chữ cái cơ sở, kết hợp 5 dấu phụ (sắc, huyền, hỏi, ngã, nặng) và 4 dấu biến âm (ă, â, ê, ô, ơ, ư, đ), tạo ra khoảng 134 ký tự unique. Điều này khiến không gian nhận dạng rộng hơn nhiều so với tiếng Anh và dễ xảy ra nhầm lẫn giữa các ký tự gần nhau về hình dạng. Thêm vào đó, các tài liệu hành chính Việt Nam thường có bố cục phức tạp: tiêu đề căn giữa, đóng dấu đè lên chữ, hai hoặc ba cột, bảng biểu đa dạng [^1].

### 2.2 PaddleOCR và kiến trúc Text Detection

PaddleOCR là bộ công cụ OCR mã nguồn mở do Baidu phát triển, nổi bật với tốc độ cao và khả năng chạy trên CPU [^2]. Trong DeepDoc, phần phát hiện văn bản (Text Detection) sử dụng mô hình DBNet (Differentiable Binarization Network) đã được chuyển sang định dạng ONNX (`det.onnx`). DBNet dự đoán xác suất nhị phân cho từng pixel thuộc vùng văn bản, sau đó áp dụng `DBPostProcess` để chuyển heatmap thành các bounding box tứ giác bao quanh từng dòng chữ.

Phiên bản mới nhất PaddleOCR 3.0 (PP-OCRv5) nâng cấp backbone lên PP-HGNetV2, áp dụng knowledge distillation từ GOT-OCR2.0 và tăng cường dữ liệu tổng hợp. Tuy nhiên do DeepDoc sử dụng model ONNX đã cố định, phiên bản cụ thể khó xác định lại; kiến trúc PP-OCRv5 được trình bày ở đây để tham chiếu kỹ thuật [^2].

### 2.3 VietOCR — Text Recognizer tiếng Việt

VietOCR [^3] là thư viện OCR mã nguồn mở chuyên biệt cho tiếng Việt, hỗ trợ hai kiến trúc:

- **vgg_seq2seq (CNN + Seq2Seq + Attention):** backbone VGG trích xuất đặc trưng ảnh, encoder GRU mã hoá chuỗi đặc trưng, decoder GRU với cơ chế attention sinh ra chuỗi ký tự. Mô hình này cân bằng tốt giữa tốc độ và độ chính xác.
- **vgg_transformer (CNN + Transformer):** thay GRU bằng Transformer encoder-decoder, lý thuyết mạnh hơn nhưng thời gian suy luận dài hơn đáng kể.

Trong hệ thống này, `vgg_seq2seq` được chọn làm backend mặc định sau khi đánh giá thực nghiệm cho thấy `vgg_transformer` không cải thiện độ chính xác tương xứng với chi phí thời gian tăng thêm.

### 2.4 ONNX và lợi ích triển khai

ONNX (Open Neural Network Exchange) là định dạng mở cho mô hình học máy, cho phép xuất và nhập mô hình giữa các framework khác nhau (PyTorch, TensorFlow, v.v.) [^4]. Lợi ích chính trong ngữ cảnh này:

- **Tương thích đa nền tảng:** ONNX Runtime hỗ trợ CPU, CUDA, DirectML, ROCm.
- **Tối ưu tốc độ:** graph optimization tự động, quantization, kernel fusion.
- **Giảm phụ thuộc:** không cần cài đặt toàn bộ framework PyTorch để inference.

Do kiến trúc VietOCR vgg_seq2seq là mô hình tự hồi quy (autoregressive), việc export ONNX cần chia thành ba graph riêng biệt: `cnn.onnx` (trích xuất đặc trưng), `encoder.onnx` (mã hoá chuỗi), và `decoder.onnx` (sinh ký tự từng bước). Phương pháp export tham khảo từ [^4].

### 2.5 YOLOv10 cho Document Layout Analysis

YOLOv10 [^5] là thế hệ mới nhất của họ YOLO, cải tiến so với YOLOv8 ở ba điểm chính: (1) kiến trúc backbone nhẹ hơn với Consistent Dual Assignments để đào tạo không cần NMS; (2) thiết kế head phân tách nhánh classification và regression (Anchor-Free decoupled head); (3) tối ưu đồng thời độ chính xác và độ trễ. Trong DeepDoc, YOLOv10 đã được fine-tune để nhận diện 10 lớp bố cục tài liệu (Layout Recognizer) và 5 thành phần cấu trúc bảng (Table Structure Recognizer), đều chạy dưới định dạng ONNX.

### 2.6 LLM trong hậu kiểm OCR

Nghiên cứu gần đây chứng minh LLM có khả năng cải thiện đáng kể chất lượng OCR thông qua post-processing, đặc biệt với các ngôn ngữ có nhiều dấu phụ [^6]. Thay vì huấn luyện lại mô hình OCR — tốn kém về dữ liệu và tài nguyên — hướng tiếp cận hậu kiểm sử dụng LLM như một tầng hiệu chuẩn ngôn ngữ: nhận văn bản OCR thô, đối chiếu với ngữ cảnh (ảnh gốc, metadata bố cục, vùng rủi ro), và trả về văn bản đã được chuẩn hóa. Điều kiện để hướng tiếp cận này khả thi trong môi trường sản xuất là mô hình LLM phải đủ nhỏ để chạy local, tránh phụ thuộc vào dịch vụ đám mây và bảo vệ dữ liệu nhạy cảm.

---

## 3. Kiến trúc hệ thống

### 3.1 Tổng quan pipeline

Hệ thống được tổ chức thành hai lớp xử lý tuần tự:

```
┌─────────────────────────────────────────────────────────────────┐
│                        TẦNG OCR CORE                            │
│                                                                  │
│  Input (PDF / Ảnh)                                               │
│       │                                                          │
│       ▼                                                          │
│  TextDetector  ──────  det.onnx (PaddleOCR DBNet)               │
│       │  bounding boxes tứ giác                                  │
│       ▼                                                          │
│  TextRecognizer ─────  VietOCR vgg_seq2seq.pth   ← mặc định     │
│       │           └──  ocr_onnx (CNN+Enc+Dec ONNX) ← tuỳ chọn   │
│       │  (text, confidence) mỗi box                              │
│       ▼                                                          │
│  LayoutRecognizer ───  layout.onnx (YOLOv10, 10 nhãn)           │
│       │  vùng: Text, Title, Table, Figure, Equation, ...         │
│       ▼                                                          │
│  TableStructureRecognizer ── tsr.onnx (YOLOv10, 5 nhãn)         │
│       │  construct_table() → Markdown bảng                       │
│       ▼                                                          │
│  Confidence Scoring + Risk Analysis + Heading Detection          │
│       │                                                          │
│       ▼                                                          │
│  *_full.md  (Markdown thô)  +  *_metadata.json                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   TẦNG HẬU KIỂM AI                               │
│                                                                  │
│  Confidence Gate ──── kiểm tra ngưỡng, quyết định gọi LLM?      │
│       │                                                          │
│       ▼ (nếu cần)                                                │
│  Correction Memory ── áp dụng rules đã học trước khi gọi LLM    │
│       │                                                          │
│       ▼                                                          │
│  LLM Post-Check ─────  OpenAI-compatible API                     │
│       │                └── Qwen3-4B (Q4_K_M) via llama.cpp/Ollama│
│       │                                                          │
│       ▼                                                          │
│  *_vlm_checked.md  +  *_vlm_review.json                          │
│       │                                                          │
│       ▼ (học hỏi)                                                │
│  Correction Memory ── cập nhật rules từ diff source↔checked      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    RAG-ready Markdown
```

Toàn bộ pipeline được điều phối bởi `full_pipeline.py`. Ba entry point phụ (`t_ocr.py`, `t_recognizer.py`) cho phép chạy từng module độc lập khi cần kiểm tra hoặc debug.

### 3.2 Text Detection — `det.onnx`

Module `TextDetector` trong `module/ocr.py` tải `det.onnx` và thực hiện ba bước: (1) tiền xử lý ảnh qua chuỗi transform trong `module/operators.py` (resize, normalize); (2) suy luận ONNX Runtime sinh heatmap xác suất văn bản; (3) hậu xử lý bằng `DBPostProcess` trong `module/postprocess.py` để chuyển heatmap thành danh sách bounding box tứ giác. Model caching được thực hiện qua dict `loaded_models` keyed bởi `(model_path, device_id)`, tránh load lại giữa các lần xử lý.

ONNX Runtime tự động chọn `CUDAExecutionProvider` khi GPU khả dụng, ngược lại dùng `CPUExecutionProvider`. Bộ nhớ GPU được giới hạn 512 MB mỗi thiết bị.

### 3.3 Text Recognition — VietOCR thay thế PaddleOCR

Đây là sự thay thế cốt lõi so với DeepDoc gốc. Thay vì dùng PaddleOCR Text Recognizer (tối ưu cho tiếng Trung/Anh), hệ thống sử dụng VietOCR với hai backend có thể hoán đổi:

**Backend PyTorch (`module/ocr.py` — mặc định):**

```python
config = Cfg.load_config_from_name('vgg_seq2seq')
config['weights'] = r"vietocr/weight/vgg_seq2seq.pth"
```

Đây là backend được khuyến nghị: độ chính xác cao hơn, xử lý tốt các dấu tiếng Việt phức tạp.

**Backend ONNX (`module/ocr_onnx.py` — tùy chọn):**

VietOCR được export sang ba graph ONNX riêng biệt do tính chất autoregressive của kiến trúc Seq2Seq. Quá trình suy luận ONNX thực hiện theo vòng lặp giải mã:

```
img → cnn.onnx → src
src → encoder.onnx → (encoder_outputs, hidden)
[sos] → decoder.onnx (+ hidden + encoder_outputs) → token[1]
token[1] → decoder.onnx → token[2] → ... → [eos]
```

Backend ONNX nhanh hơn nhưng độ chính xác giảm nhẹ do lượng tử hóa và sai số tích lũy qua nhiều bước giải mã.

Hai backend có interface giống nhau (`OCR` class), cho phép hoán đổi chỉ bằng cách đổi dòng import mà không cần thay đổi code pipeline.

### 3.4 Layout Recognizer — YOLOv10 (10 nhãn)

`LayoutRecognizer4YOLOv10` trong `module/layout_recognizer.py` chạy `layout.onnx` để phân vùng mỗi trang thành 10 loại vùng nội dung:

| Nhãn | Ý nghĩa |
|---|---|
| Text | Đoạn văn bản thông thường |
| Title | Tiêu đề |
| Image | Hình ảnh |
| Image Caption | Chú thích hình |
| Table | Bảng dữ liệu |
| Table Caption | Chú thích bảng |
| Header | Đầu trang |
| Footer | Chân trang |
| Reference | Tài liệu tham khảo |
| Equation | Công thức toán học |

Kết quả phân vùng được dùng để: (1) định hướng thứ tự đọc đúng (đặc biệt quan trọng với tài liệu nhiều cột); (2) quyết định vùng nào đưa vào TSR; (3) phát hiện vùng Figure/Equation để lưu ảnh crop thay vì cố OCR.

### 3.5 Table Structure Recognizer — YOLOv10 (5 nhãn)

`TableStructureRecognizer` chạy `tsr.onnx` trên ảnh crop của từng vùng Table, nhận dạng 5 thành phần:

| Nhãn | Ý nghĩa |
|---|---|
| table column | Cột |
| table row | Hàng |
| table column header | Đầu đề cột |
| table projected row header | Đầu đề hàng chiếu |
| table spanning cell | Ô trải dài |

Sau khi có các thành phần, `construct_table()` thực hiện logic gán ô: với mỗi box OCR trong vùng bảng, tìm Row (R), Column (C), Header (H) và Spanning cell (SP) bằng cách kiểm tra overlap. Kết quả được serialize thành Markdown table chuẩn, bảo toàn cấu trúc hàng/cột khi đưa vào RAG.

### 3.6 OCR Confidence Scoring

Mỗi dòng văn bản được VietOCR trả về kèm điểm tin cậy (confidence score) trong khoảng [0, 1]. Hệ thống tổng hợp thống kê theo ba cấp độ:

**Cấp region:** tổng hợp tất cả dòng trong một vùng bố cục.

**Cấp trang:** tổng hợp theo weighted average từ các regions.

**Cấp document:**

```
avg_score   = Σ(avg_region × line_count_region) / Σ(line_count_region)
min_score   = min toàn document
median_score= trung vị tất cả dòng
low_score_ratio = số dòng có score < threshold / tổng số dòng
```

Tương tự, **Layout Confidence Scoring** tổng hợp detection score của từng region từ YOLOv10.

Toàn bộ số liệu được ghi vào `*_metadata.json` để phục vụ cả quyết định kích hoạt VLM lẫn truy vết sau này.

### 3.7 VLM Risk Analysis

Trước khi gọi LLM, hệ thống quét toàn bộ văn bản OCR để phát hiện các vùng cần ưu tiên kiểm tra (VLM risks). Mỗi risk được gán `risk_id`, tọa độ bbox, severity và ảnh crop tương ứng để đính kèm vào prompt LLM.

Bốn loại rủi ro được nhận diện qua regex:

| Loại rủi ro | Ví dụ điển hình | Severity |
|---|---|---|
| `sensitive_value` | Số tiền, ngày tháng, CCCD, mã hợp đồng | HIGH |
| `ambiguous_characters` | Token chứa O/0, l/1, S/5 lẫn lộn | HIGH |
| `checkbox_or_selection` | `[x]`, `☑`, `□`, "chọn" | MEDIUM |
| `ocr_too_short_for_region` | Văn bản ngắn bất thường so với diện tích vùng | MEDIUM |

Ngoài ra, hệ thống phát hiện **bố cục nhiều cột** (`page_has_multicolumn_layout`) dựa trên phân phối tâm của các region để cảnh báo nguy cơ sai thứ tự đọc.

Các ảnh crop vùng rủi ro được lưu vào thư mục `assets/` theo tên `page_NNN_risk_NNN_<category>.png` và tham chiếu trong metadata để đính kèm vào prompt VLM.

### 3.8 Nhận diện cấu trúc văn bản pháp lý tiếng Việt (Heading Detection)

Một thách thức đặc thù của tài liệu hành chính Việt Nam là cấu trúc heading không được đánh dấu tường minh trong OCR raw. Hệ thống áp dụng pattern matching để phát hiện và gán cấp heading Markdown tự động, tuân theo quy ước văn bản pháp lý:

| Pattern | Cấp heading | Ví dụ |
|---|---|---|
| Tên loại văn bản | `#` (H1) | "Quyết định", "Thông báo", "Hợp đồng" |
| Điều khoản, Nơi nhận | `##` (H2) | "Điều 1. Phạm vi áp dụng" |
| Chương/Phần | `##`–`###` (H2–H3) | "Chương II", "Phần A", "Mục 3" |
| Mục số cấp 1 | `###` (H3) | "1. Mục tiêu" |
| Mục số cấp 2 | `####` (H4) | "1.2. Tiến độ thực hiện" |

Hệ thống duy trì `state` xuyên suốt tài liệu để xử lý đúng trường hợp "dense subsection" — mục con được viết liền (`12. Mục` thay vì `1.2. Mục`) do giới hạn nhận dạng OCR.

### 3.9 AI Post-Check — Hậu kiểm LLM

#### 3.9.1 Mô hình ngôn ngữ và giao thức giao tiếp

Hệ thống sử dụng **Qwen3-4B** (quantized Q4_K_M) chạy local thông qua **llama.cpp** hoặc **Ollama**, giao tiếp qua giao thức **OpenAI-compatible API**. Quyết định thiết kế này mang lại hai lợi thế quan trọng: (1) dữ liệu tài liệu nội bộ không rời khỏi hạ tầng cục bộ; (2) có thể hoán đổi mô hình hoặc backend mà không thay đổi code.

#### 3.9.2 Cơ chế kích hoạt thích nghi (Confidence Gate)

Hậu kiểm LLM không chạy vô điều kiện trên mọi tài liệu. Hệ thống hỗ trợ 4 chế độ kích hoạt cấu hình qua biến môi trường `VLM_POSTCHECK_TRIGGER`:

| Chế độ | Hành vi |
|---|---|
| `always` | Luôn gọi LLM, bất kể confidence |
| `never` | Bỏ qua hoàn toàn tầng LLM |
| `low_confidence` | Gọi LLM khi bất kỳ ngưỡng nào bị vi phạm (mặc định) |
| `risk_or_low_confidence` | Gọi LLM nếu có risk hoặc confidence thấp |

Chế độ `low_confidence` đánh giá tổng hợp nhiều tiêu chí:

- `avg_score < VLM_OCR_AVG_CONFIDENCE_THRESHOLD`
- `min_score < VLM_OCR_VERY_LOW_LINE_THRESHOLD`
- `low_score_ratio ≥ VLM_OCR_LOW_LINE_RATIO`
- `layout_avg_score < VLM_LAYOUT_AVG_CONFIDENCE_THRESHOLD`
- Số dòng OCR quá ít so với vùng text/table được nhận diện

#### 3.9.3 System Prompt và chiến lược prompt engineering

System prompt (`conf/ai_postcheck_system_prompt.txt`) được thiết kế theo nguyên tắc **trung thành với tài liệu gốc là ưu tiên tối thượng**, phân chia rõ ràng thành các nhóm quy tắc:

**Được phép sửa (khi có bằng chứng trực tiếp từ ảnh/crop):**
- Lỗi dấu tiếng Việt, ký tự nhầm (O/0, I/l/1, S/5), khoảng trắng, xuống dòng
- Cấu trúc bảng Markdown bị lệch hàng/cột
- Thứ tự đọc khi OCR trộn cột
- Trạng thái checkbox nếu nhìn thấy rõ

**Nghiêm cấm:**
- Thay tên tổ chức/cơ quan/người dù "có vẻ đúng hơn" theo suy luận
- Chuẩn hóa tên theo phiên bản hiện hành, rebrand, sáp nhập
- Suy luận và bịa thêm số liệu vào bảng
- Với tiêu đề, con dấu, header/footer, tên bên ký: mức bảo toàn cao nhất

Nguyên tắc "khi nghi ngờ, giữ nguyên và ghi vào `issues`" giúp tránh LLM hallucination trong những trường hợp mơ hồ — đặc biệt quan trọng với tài liệu pháp lý, hợp đồng.

#### 3.9.4 Schema đầu ra có cấu trúc

LLM được yêu cầu trả về JSON hợp lệ duy nhất theo schema cố định:

| Trường | Nội dung |
|---|---|
| `checked_markdown` | Toàn bộ Markdown sau hậu kiểm |
| `summary` | Tóm tắt ngắn những thay đổi đã thực hiện |
| `issues` | Danh sách vấn đề nghi ngờ hoặc cần đối chiếu thủ công |
| `image_notes` | Mô tả ngắn các crop quan trọng (chữ ký, QR, con dấu) |
| `vlm_findings` | Kết quả kiểm tra theo `risk_id` |
| `confidence` | Điểm tin cậy tổng thể [0–1] |

Output artifacts trên đĩa: `*_vlm_checked.md` (markdown đã kiểm tra) và `*_vlm_review.json` (toàn bộ payload JSON để truy vết).

### 3.10 Correction Memory

Correction Memory là hệ thống học tăng dần (incremental learning) từ kết quả hậu kiểm, lưu trữ tại `conf/ocr_correction_memory.json`.

**Luồng học (Learning):** sau mỗi lần LLM post-check, hệ thống so sánh diff giữa markdown thô và markdown đã kiểm tra bằng `difflib.SequenceMatcher`. Mỗi cặp thay đổi `(wrong, correct)` được đánh giá qua bộ lọc:

- Độ dài: không quá 240 ký tự hoặc 4 dòng
- Không phải nội dung có cấu trúc (table HTML, link ảnh)
- Similarity ratio ≥ 0.72 (đảm bảo là sửa lỗi, không phải viết lại hoàn toàn)

Các cặp vượt qua bộ lọc được lưu thành rule có `id` (SHA-256 hash của cặp), `seen_count`, `applied_count`, `last_applied_at`.

**Luồng áp dụng (Application):** ở đầu mỗi lần chạy pipeline, correction memory được nạp và áp dụng trên markdown thô theo thứ tự từ rule dài nhất trước (tránh xung đột). Rules bị block (người dùng từ chối qua Web UI) được bỏ qua.

**Tích hợp Web UI:** người dùng có thể xem từng thay đổi LLM đã thực hiện, chấp nhận hoặc từ chối (reverse) từng thay đổi riêng lẻ. Khi từ chối một thay đổi, rule tương ứng trong correction memory bị đánh dấu `blocked`, ngăn áp dụng lại trong tương lai.

---

## 4. Triển khai

### 4.1 Cài đặt local

```bash
git clone https://github.com/nhnnhnnnhn/deepdoc_vietocr.git
cd deepdoc_vietocr
pip install -r requirements.txt
```

Đặt file trọng số VietOCR (tải từ trang chính thức VietOCR):
```
vietocr/weight/vgg_seq2seq.pth
```

Các file ONNX (`det.onnx`, `layout.onnx`, `tsr.onnx`) tự động tải từ HuggingFace `InfiniFlow/deepdoc` lần đầu chạy. Trên môi trường không có kết nối HuggingFace trực tiếp (phổ biến tại Việt Nam):
```bash
set HF_ENDPOINT=https://hf-mirror.com
```

Cấu hình LLM post-check trong file `.env`:
```
GEMINI_POSTCHECK=true
OPENAI_API_BASE=http://localhost:8080/v1   # llama.cpp server
OPENAI_API_KEY=local
OPENAI_MODEL_NAME=qwen3-4b
VLM_POSTCHECK_TRIGGER=low_confidence
```

**Các lệnh entry point:**

```bash
# OCR thuần: output ảnh annotated + file .txt
python t_ocr.py --inputs=<path> --output_dir=./ocr_outputs

# Layout recognition: output ảnh annotated
python t_recognizer.py --inputs=<path> --threshold=0.2 --mode=layout

# Table structure recognition: output ảnh annotated + .md
python t_recognizer.py --inputs=<path> --threshold=0.2 --mode=tsr

# Full pipeline: layout + TSR + OCR + AI post-check → .md
python full_pipeline.py --inputs=<path> --output_dir=./output --threshold=0.5
```

> **Lưu ý:** tất cả script chuyển stdout/stderr sang file log trong `log/`. Kiểm tra `log/full_pipeline.log` để theo dõi tiến trình.

### 4.2 Web Application

`web_app.py` cung cấp REST API và giao diện web được xây dựng bằng FastAPI, cho phép:

- **Upload tài liệu** (PDF, PNG, JPG, TIFF, BMP) và chạy full pipeline trong background
- **Theo dõi tiến trình** qua live log streaming (polling offset-based)
- **Xem và tải kết quả:** markdown đã kiểm tra, metadata JSON
- **Review thay đổi AI:** xem danh sách từng thay đổi LLM đã thực hiện kèm severity và page number, chấp nhận hoặc từ chối từng thay đổi riêng lẻ
- **Lịch sử job:** lưu persistent qua `history.json`, phục hồi sau restart

```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000
# Truy cập: http://localhost:8000
```

### 4.3 Docker

Hệ thống cung cấp Docker image cho hai loại GPU:

**NVIDIA CUDA:**
```bash
cp .env.docker.example .env
docker compose build
docker compose up
```

**AMD ROCm 7:**
```bash
cp .env.rocm.example .env
docker compose -f docker-compose.rocm.yml build
docker compose -f docker-compose.rocm.yml up
```

Lần đầu chạy AMD ROCm, MIGraphX compile các ONNX model — request đầu tiên có thể chậm hơn 30–90 giây.

---

## 5. Đánh giá và thảo luận

### 5.1 So sánh các backend OCR

Ba lựa chọn backend được cung cấp với trade-off tốc độ/độ chính xác khác nhau:

| Backend | Tốc độ | Độ chính xác dấu TV | Ghi chú |
|---|---|---|---|
| VietOCR PyTorch `vgg_seq2seq` | Trung bình | Cao | **Khuyến nghị mặc định** |
| VietOCR PyTorch `vgg_transformer` | Chậm | Tương đương seq2seq | Không đề xuất — chi phí cao, lợi ích thấp |
| VietOCR ONNX (3 graph) | Nhanh nhất | Thấp hơn nhẹ | Dùng khi ưu tiên throughput |

Với pipeline có AI post-check, backend ONNX trở nên hợp lý hơn vì LLM sẽ bù đắp lỗi nhỏ của OCR. Tuy nhiên với tài liệu có nhiều số liệu quan trọng (sensitive_value), nên giữ backend PyTorch để giảm nguy cơ nhầm O/0, l/1 trước khi vào tầng LLM.

### 5.2 Hiệu quả AI Post-Check

Tầng hậu kiểm LLM giải quyết hiệu quả các vấn đề mà OCR thuần không thể tự xử lý:

- **Sửa dấu tiếng Việt theo ngữ cảnh:** thay vì chỉ nhận dạng ký tự đơn lẻ, LLM hiểu ngữ nghĩa cả câu để xác định dấu đúng.
- **Chuẩn hóa cấu trúc Markdown:** tinh chỉnh heading, bảng, danh sách theo ngữ cảnh tài liệu.
- **Phát hiện và ghi nhận vùng mơ hồ:** thay vì đoán mò, hệ thống ghi `cần đối chiếu thủ công` vào `issues`, bảo toàn tính trung thực của tài liệu.

Cơ chế confidence-gated đảm bảo chi phí LLM chỉ phát sinh khi thực sự cần thiết. Với tài liệu chất lượng scan tốt, pipeline có thể bỏ qua tầng LLM hoàn toàn và vẫn đạt kết quả đủ tốt cho RAG.

### 5.3 Hạn chế và hướng cải thiện

**Hạn chế hiện tại:**

- **Chưa có đánh giá định lượng chuẩn:** do chưa xây dựng tập ground truth riêng, các nhận xét về độ chính xác mang tính định tính.
- **Phụ thuộc chất lượng scan:** với ảnh có độ phân giải thấp (< 150 DPI) hoặc bị nghiêng nhiều, cả OCR và LLM đều giảm hiệu quả đáng kể.
- **Tốc độ tầng LLM:** với tài liệu nhiều trang, thời gian post-check tỷ lệ tuyến tính với số trang, có thể là bottleneck.
- **Correction Memory chưa có cơ chế phân rã:** rules ít được dùng không tự động xóa, có thể gây nhiễu theo thời gian.

**Hướng cải thiện đề xuất:**

- Xây dựng tập benchmark gồm tài liệu tiếng Việt với ground truth để đo CER/WER định lượng.
- Tích hợp tiền xử lý ảnh (deskewing, super-resolution) để cải thiện đầu vào OCR với tài liệu chất lượng thấp.
- Batching các trang vào một lời gọi LLM duy nhất để giảm overhead.
- Thêm cơ chế TTL cho correction memory rules.

---

## 6. Kết luận và hướng phát triển

### 6.1 Tổng kết đóng góp

Bài báo cáo trình bày một pipeline trích xuất tài liệu tiếng Việt hoàn chỉnh, kết hợp ba lớp công nghệ: OCR nhận dạng ký tự (VietOCR), phân tích bố cục (YOLOv10), và hậu kiểm ngôn ngữ (Qwen3-4B local LLM). Hệ thống được thiết kế với triết lý "privacy-first, cost-efficient": toàn bộ suy luận chạy local, không phụ thuộc API đám mây, hoạt động trên CPU mà vẫn đạt hiệu năng chấp nhận được.

Các đóng góp kỹ thuật cụ thể bao gồm: cơ chế confidence-gated để kiểm soát chi phí LLM, hệ thống phát hiện rủi ro trước hậu kiểm, nhận diện cấu trúc văn bản pháp lý tiếng Việt, correction memory học tăng dần, và giao diện web với khả năng review từng thay đổi AI. Kết quả cuối cùng là file Markdown có cấu trúc, đúng chính tả, sẵn sàng đưa vào chunking và vector store của hệ thống RAG.

### 6.2 Hướng phát triển tiếp theo

Bước tiếp theo của dự án là tích hợp pipeline OCR này vào hệ thống RAG + Chatbot hoàn chỉnh:

- **Document ingestion:** đưa markdown output vào pipeline chunking (semantic hoặc fixed-size) và embedding vào vector store (Chroma, Qdrant, hoặc tương đương).
- **Retrieval layer:** thiết kế retriever kết hợp dense và sparse search để tối ưu truy xuất tài liệu pháp lý có cấu trúc đặc thù.
- **Chatbot interface:** xây dựng giao diện hỏi-đáp với khả năng trích dẫn nguồn (source attribution) đến trang và vùng cụ thể trong tài liệu gốc, tận dụng metadata đã thu thập trong pipeline OCR.

---

## 7. Tài liệu tham khảo

[^1]: A Survey on Vietnamese Document Analysis and Recognition: Challenges and Future Directions. Moonlight AI, 2026. https://www.themoonlight.io/en/review/a-survey-on-vietnamese-document-analysis-and-recognition-challenges-and-future-directions

[^2]: Cui, C., Sun, T., Lin, M., et al. PaddleOCR 3.0 Technical Report. arXiv:2507.05595, 2025. https://arxiv.org/html/2507.05595v1

[^3]: Quoc, P.B. VietOCR: A Transformer-based Vietnamese Text Recognition. GitHub, 2021. https://github.com/pbcquoc/vietocr

[^4]: Chuyển đổi mô hình học sâu về ONNX — VietOCR sang ONNX. Viblo, 2021. https://viblo.asia/p/chuyen-doi-mo-hinh-hoc-sau-ve-onnx-bWrZnz4vZxw

[^5]: Wang, A., Chen, H., Liu, L., et al. YOLOv10: Real-Time End-to-End Object Detection. NeurIPS 2024. arXiv:2405.14458. https://arxiv.org/pdf/2405.14458

[^6]: Reference-Based Post-OCR Processing with LLM for Precise Diacritic Text in Historical Document Recognition. arXiv:2410.13305, 2024. https://arxiv.org/html/2410.13305v3

[^7]: InfiniFlow. DeepDoc: Document Understanding for RAGFlow. GitHub, 2024. https://github.com/infiniflow/ragflow/blob/main/deepdoc/README.md

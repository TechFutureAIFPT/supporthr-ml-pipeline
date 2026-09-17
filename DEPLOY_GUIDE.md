# SupportHR Classifier Microservice (Kaggle & Colab & Cloud)

Dịch vụ độc lập phục vụ phân loại ngành nghề CV (24 nhóm ngành nghề chuẩn SupportHR).
Được thiết kế để chạy độc lập tách rời hoàn toàn khỏi `cv-match-api`, có thể triển khai trên Google Colab, Kaggle Notebooks, Docker hoặc Cloud VPS.

---

## 1. Cấu Trúc Dịch Vụ

```
Software/backend/classifier-service/
├── server.py              # FastAPI microservice độc lập (cổng 8000/8123)
├── colab_classifier.ipynb # Notebook 1-Click triển khai trên Google Colab (GPU T4 miễn phí)
├── kaggle_classifier.ipynb# Notebook 1-Click triển khai trên Kaggle (gắn tập dữ liệu Resume.csv)
├── run_live_tunnel.py     # Script chạy live server + mở Cloudflare Tunnel + cập nhật .env
├── test_connection.py     # Script kiểm tra kết nối và độ tương thích hợp đồng API
├── requirements.txt       # Danh sách thư viện tối giản
└── README.md              # Hướng dẫn chi tiết
```

---

## 2. Các Endpoint Cung Cấp

| Endpoint | Method | Payload / Response |
| :--- | :--- | :--- |
| `/api/cv/classify-industry` | `POST` | Body: `{"cv_text": "...", "top_k": 3}`<br>Trực tiếp dự đoán ngành nghề phù hợp nhất. |
| `/api/cv/classifier-status` | `GET` | Trả về trạng thái `ready: true/false`, danh sách 24 nhãn ngành và nguồn model. |
| `/health` | `GET` | Health check nhanh `{"status": "ok"}`. |

---

## 3. Cách Sử Dụng

### Cách 1: Chạy Live Tunnel Cục Bộ (Tự động cập nhật `api_server/.env`)
```bash
python run_live_tunnel.py
```
Script sẽ tự động khởi động `server.py`, mở Cloudflare Tunnel HTTPS và cập nhật các biến `LOCAL_CLASSIFIER_*` vào `Software/backend/cv-match-api/api_server/.env`.

### Cách 2: Triển khai trên Google Colab
1. Mở [Google Colab](https://colab.research.google.com/) > Upload file `colab_classifier.ipynb`.
2. Chọn **Runtime** > **Run all**.
3. Lấy URL Cloudflare HTTPS in ra ở cuối notebook.

### Cách 3: Triển khai trên Kaggle Notebooks
1. Mở [Kaggle](https://www.kaggle.com/) > **Code** > **New Notebook** > Import file `kaggle_classifier.ipynb`.
2. Gắn dataset `resume dataset` (tùy chọn để huấn luyện trên dữ liệu thật).
3. Chọn **Run all** và lấy URL HTTPS.

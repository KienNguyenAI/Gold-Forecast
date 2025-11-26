#  Gold Price Forecast System

Hệ thống dự báo giá vàng sử dụng mô hình Hybrid (LSTM + Macroeconomics Data).

## Tính năng
- **Tự động tải dữ liệu:** Yahoo Finance & FRED API.
- **Mô hình Hybrid:** Kết hợp phân tích kỹ thuật (Technical) và vĩ mô (Macro).
- **Backtesting:** Kiểm thử chiến lược đầu tư so với Buy & Hold.
- **Dự báo:** Đưa ra vùng giá Min/Max cho 30 ngày tới.

##  Cài đặt
1. Clone repo
2. `pip install -r requirements.txt`
3. Tạo file `.env` và điền `FRED_API_KEY=...`
4. Chạy: `python main.py pipeline`
## Đánh giá 
- **Evaluator.py** = Chỉ dùng để đo Accuracy. KHÔNG dùng để tính P/L.
- **Backtester.py** = Đánh giá hiệu suất thật. Dùng kết quả này để quyết định chiến lược.
Chỉ cần Accuracy > 55% là mô hình có thể tạo lợi nhuận với quản lý vốn hợp lý.

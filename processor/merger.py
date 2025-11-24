import pandas as pd
import numpy as np
import os


class DataMerger:
    def __init__(self, raw_data_dir="data/raw"):
        self.raw_dir = raw_data_dir

    def load_and_merge(self):
        print("🔄 Đang xử lý và ghép dữ liệu...")

        # --- BƯỚC 1: Load Dữ liệu Vàng (Backbone) ---
        gold_path = os.path.join(self.raw_dir, "Gold_daily.csv")
        if not os.path.exists(gold_path):
            raise FileNotFoundError("Chưa chạy main_fetch.py hoặc thiếu file Gold!")

        df_gold = pd.read_csv(gold_path, index_col=0, parse_dates=True)
        # Chỉ giữ lại các cột quan trọng
        # Kiểm tra xem cột Close có tồn tại không để tránh lỗi case-sensitive
        if 'Close' in df_gold.columns:
            df_gold = df_gold[['Close', 'Volume']]
        elif 'Adj Close' in df_gold.columns:
            df_gold = df_gold[['Adj Close', 'Volume']]

        df_gold.columns = ['Gold_Close', 'Gold_Volume']

        # --- BƯỚC 2: Load & Ghép Dữ liệu Vĩ mô ---
        macro_files = {
            'DXY': 'DXY_daily.csv',  # DXY có 5 cột (Open, High, Low, Close, Volume)
            'US10Y': 'US10Y_macro.csv',  # FRED: 1 cột
            'CPI': 'CPI_macro.csv',  # FRED: 1 cột
            'Real_Rate': 'Real_Interest_Rate_macro.csv'  # FRED: 1 cột
        }

        for name, filename in macro_files.items():
            path = os.path.join(self.raw_dir, filename)
            if os.path.exists(path):
                df_macro = pd.read_csv(path, index_col=0, parse_dates=True)

                # --- [FIX LỖI TẠI ĐÂY] ---
                # Nếu file có nhiều cột (như DXY), chỉ lấy cột Close
                if len(df_macro.columns) > 1:
                    if 'Close' in df_macro.columns:
                        df_macro = df_macro[['Close']]
                    elif 'Adj Close' in df_macro.columns:
                        df_macro = df_macro[['Adj Close']]
                    else:
                        # Fallback: Nếu không tìm thấy Close, lấy cột đầu tiên
                        df_macro = df_macro.iloc[:, [0]]

                # Giờ thì df_macro chắc chắn chỉ còn 1 cột, đổi tên thoải mái
                df_macro.columns = [name]

                # GHÉP: Left Join vào bảng Gold theo Index (Date)
                df_gold = df_gold.join(df_macro, how='left')

                # QUAN TRỌNG: Forward Fill
                df_gold[name] = df_gold[name].ffill()
            else:
                print(f"⚠️ Cảnh báo: Không tìm thấy file {filename}")

        # Xóa các dòng đầu tiên bị NaN do chưa có dữ liệu vĩ mô
        df_gold.dropna(inplace=True)
        return df_gold

    def create_targets(self, df, prediction_window=30):

        print(f"🎯 Đang tạo nhãn dự báo cho {prediction_window} ngày tới...")

        # Logic Rolling Window hướng về tương lai
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=prediction_window)

        future_min = df['Gold_Close'].rolling(window=indexer).min()
        future_max = df['Gold_Close'].rolling(window=indexer).max()

        # Chuyển đổi sang % thay đổi (Relative Change)
        # Để mô hình học được biên độ thay vì học giá tiền
        df['Target_Min_Change'] = (future_min - df['Gold_Close']) / df['Gold_Close']
        df['Target_Max_Change'] = (future_max - df['Gold_Close']) / df['Gold_Close']

        # Xóa các dòng cuối cùng (vì không đủ 30 ngày tương lai để tính target)
        df = df.dropna()

        return df
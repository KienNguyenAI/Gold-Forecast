import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib  # Để lưu lại bộ Scaler dùng cho sau này
import os

class DataProvider:
    def __init__(self, data_path=None, window_size=60):
        if data_path is None:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))

            # Cách 1: Thử lùi 1 cấp (Giả sử cấu trúc là Gold/training/file.py)
            path_option_1 = os.path.join(os.path.dirname(current_file_dir), "data", "processed", "Master_Dataset.csv")

            # Cách 2: Thử lùi 2 cấp (Giả sử cấu trúc là Gold/src/training/file.py)
            path_option_2 = os.path.join(os.path.dirname(os.path.dirname(current_file_dir)), "data", "processed",
                                         "Master_Dataset.csv")

            # Kiểm tra xem cái nào đúng
            if os.path.exists(path_option_1):
                self.data_path = path_option_1
            elif os.path.exists(path_option_2):
                self.data_path = path_option_2
            else:
                # Nếu không tìm thấy cả 2, mặc định dùng option 1 để báo lỗi cho dễ hiểu
                self.data_path = path_option_1
        else:
            self.data_path = data_path

        print(f"📂 Đang đọc dữ liệu từ: {self.data_path}")

        # Kiểm tra lần cuối
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"❌ Vẫn không tìm thấy file! Bạn hãy kiểm tra xem file 'Master_Dataset.csv' có nằm trong thư mục 'data/processed' không?")
        self.window_size = window_size

        # Định nghĩa các cột sẽ dùng cho từng nhánh
        # Nhánh 1: LSTM (Technical)
        self.tech_cols = ['Gold_Close', 'Log_Return', 'RSI', 'Volatility_20d', 'Trend_Signal']

        # Nhánh 2: Dense (Macro) - Lưu ý tên cột phải khớp với file CSV của bạn
        # Nếu file CSV dùng tên khác (ví dụ US10Y thay vì DGS10), hãy sửa ở đây
        self.macro_cols = ['DXY', 'US10Y', 'CPI', 'Real_Rate']

        # Target (Output)
        self.target_cols = ['Target_Min_Change', 'Target_Max_Change']

        # Scaler
        self.scaler_tech = MinMaxScaler()
        self.scaler_macro = MinMaxScaler()

    def load_and_split(self, train_ratio=0.8):
        """
        Hàm chính để chuẩn bị dữ liệu train/test
        """
        # 1. Load dữ liệu
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)

        # Kiểm tra xem các cột macro có tồn tại không, nếu thiếu thì bỏ qua hoặc báo lỗi
        available_macro = [c for c in self.macro_cols if c in df.columns]
        if len(available_macro) < len(self.macro_cols):
            print(f"⚠️ Cảnh báo: Thiếu cột Macro. Tìm thấy: {available_macro}")
            self.macro_cols = available_macro

        # 2. Chuẩn hóa (Scaling) - Cực kỳ quan trọng cho LSTM
        # Fit scaler trên toàn bộ data (hoặc chỉ train set để chuẩn xác hơn, nhưng ở đây làm đơn giản trước)
        data_tech_scaled = self.scaler_tech.fit_transform(df[self.tech_cols])
        data_macro_scaled = self.scaler_macro.fit_transform(df[self.macro_cols])
        targets = df[self.target_cols].values  # Target % change thường nhỏ nên không cần scale, hoặc scale tùy ý

        # 3. Tạo Sliding Window (Cắt lát dữ liệu)
        X_tech, X_macro, y = [], [], []

        # Chạy từ ngày thứ 60 đến hết
        for i in range(self.window_size, len(df)):
            # Input A: Lấy 60 ngày quá khứ của các chỉ số kỹ thuật
            tech_window = data_tech_scaled[i - self.window_size:i]

            # Input B: Lấy giá trị vĩ mô của ngày hiện tại (ngày thứ i)
            # Lý do: Ta muốn biết vĩ mô HÔM NAY ảnh hưởng thế nào đến tương lai
            macro_current = data_macro_scaled[i - 1]

            # Output: Target của dòng hiện tại
            target_current = targets[i]

            X_tech.append(tech_window)
            X_macro.append(macro_current)
            y.append(target_current)

        # Chuyển sang Numpy Array
        X_tech = np.array(X_tech)
        X_macro = np.array(X_macro)
        y = np.array(y)

        # 4. Chia Train / Test
        split_idx = int(len(X_tech) * train_ratio)

        X_train = {
            'input_price': X_tech[:split_idx],
            'input_macro': X_macro[:split_idx]
        }
        y_train = {
            'output_min': y[:split_idx, 0],
            'output_max': y[:split_idx, 1]
        }

        X_test = {
            'input_price': X_tech[split_idx:],
            'input_macro': X_macro[split_idx:]
        }
        y_test = {
            'output_min': y[split_idx:, 0],
            'output_max': y[split_idx:, 1]
        }

        print(f"✅ Đã chuẩn bị dữ liệu xong!")
        print(f"   - Shape X_tech (Train): {X_train['input_price'].shape}")
        print(f"   - Shape X_macro (Train): {X_train['input_macro'].shape}")

        return X_train, y_train, X_test, y_test

    def save_scalers(self, path="src/models/"):
        """Lưu scaler để dùng lúc dự đoán thực tế (Inference)"""
        joblib.dump(self.scaler_tech, f"{path}scaler_tech.pkl")
        joblib.dump(self.scaler_macro, f"{path}scaler_macro.pkl")
        print("💾 Đã lưu Scalers.")


# --- Test thử ---
if __name__ == "__main__":
    provider = DataProvider(window_size=60)
    try:
        X_train, y_train, X_test, y_test = provider.load_and_split()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
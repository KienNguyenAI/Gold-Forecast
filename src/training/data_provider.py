import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import logging
from typing import Dict, Tuple


class DataProvider:
    def __init__(self, settings: Dict):
        """
        Khởi tạo DataProvider với cấu hình.
        :param settings: Config dictionary (settings.yaml)
        """
        self.logger = logging.getLogger(__name__)
        self.settings = settings

        # Lấy đường dẫn từ config
        processed_dir = settings['paths']['processed_data']
        self.data_path = os.path.join(processed_dir, "gold_processed_features.csv")
        self.model_save_path = settings['paths']['model_save']  # Nơi lưu scaler

        # Lấy tham số xử lý
        self.window_size = settings['processing']['window_size']
        self.test_ratio = settings['processing']['test_size']

        # Định nghĩa cột (Có thể đưa vào config nếu muốn linh hoạt hơn nữa)
        self.tech_cols = ['Gold_Close', 'Log_Return', 'RSI', 'Volatility_20d', 'Trend_Signal']
        self.macro_cols = ['DXY', 'US10Y', 'CPI', 'Real_Rate']
        self.target_cols = ['Target_Min_Change', 'Target_Max_Change']

        self.scaler_tech = MinMaxScaler()
        self.scaler_macro = MinMaxScaler()

    def load_and_split(self) -> Tuple[Dict, Dict, Dict, Dict]:
        self.logger.info(f"📂 Đang đọc dữ liệu từ: {self.data_path}")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu tại: {self.data_path}")

        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)

        # Kiểm tra cột thiếu
        missing_tech = [c for c in self.tech_cols if c not in df.columns]
        missing_macro = [c for c in self.macro_cols if c not in df.columns]

        if missing_tech or missing_macro:
            self.logger.warning(f"⚠️ Thiếu cột dữ liệu! Tech: {missing_tech}, Macro: {missing_macro}")
            # Ở đây có thể raise error nếu muốn nghiêm ngặt

        # Scaling
        # Lưu ý: Nên split trước khi scale để tránh data leakage,
        # nhưng ở bước này mình giữ nguyên logic cũ của bạn cho đơn giản.
        data_tech_scaled = self.scaler_tech.fit_transform(df[self.tech_cols])
        data_macro_scaled = self.scaler_macro.fit_transform(df[self.macro_cols])
        targets = df[self.target_cols].values

        # Sliding Window
        X_tech, X_macro, y = [], [], []

        # Logic tạo window
        for i in range(self.window_size, len(df)):
            tech_window = data_tech_scaled[i - self.window_size:i]
            macro_current = data_macro_scaled[i - 1]  # Macro tại thời điểm t-1
            target_current = targets[i]

            X_tech.append(tech_window)
            X_macro.append(macro_current)
            y.append(target_current)

        X_tech = np.array(X_tech)
        X_macro = np.array(X_macro)
        y = np.array(y)

        # Split Train/Test
        split_idx = int(len(X_tech) * (1 - self.test_ratio))  # Ví dụ 0.8

        X_train = {'input_price': X_tech[:split_idx], 'input_macro': X_macro[:split_idx]}
        y_train = {'output_min': y[:split_idx, 0], 'output_max': y[:split_idx, 1]}

        X_test = {'input_price': X_tech[split_idx:], 'input_macro': X_macro[split_idx:]}
        y_test = {'output_min': y[split_idx:, 0], 'output_max': y[split_idx:, 1]}

        self.logger.info(
            f"✅ Đã split dữ liệu. Train size: {len(X_tech[:split_idx])}, Test size: {len(X_tech[split_idx:])}")

        return X_train, y_train, X_test, y_test

    def save_scalers(self):
        """Lưu scaler vào artifacts/models/"""
        os.makedirs(self.model_save_path, exist_ok=True)
        joblib.dump(self.scaler_tech, os.path.join(self.model_save_path, "scaler_tech.pkl"))
        joblib.dump(self.scaler_macro, os.path.join(self.model_save_path, "scaler_macro.pkl"))
        self.logger.info(f"💾 Đã lưu Scalers vào: {self.model_save_path}")
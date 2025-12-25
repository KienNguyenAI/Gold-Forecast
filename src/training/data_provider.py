import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import logging
from typing import Dict, Tuple


class DataProvider:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.settings = settings

        processed_dir = settings['paths']['processed_data']
        self.data_path = os.path.join(processed_dir, "gold_processed_features.csv")
        self.model_save_path = settings['paths']['model_save']

        # Lấy tham số từ settings (lưu ý settings này sẽ được trainer cập nhật động)
        self.window_size = settings['processing'].get('window_size', 60)
        self.test_ratio = settings['processing'].get('test_size', 0.2)

        self.tech_cols = ['Gold_Close', 'Log_Return', 'RSI', 'Volatility_20d', 'Trend_Signal']
        self.macro_cols = ['DXY', 'US10Y', 'CPI', 'Real_Rate']

        horizon = settings.get('model', {}).get('forecast_horizon', 30)
        self.target_cols = [f'Target_Min_Change_{horizon}', f'Target_Max_Change_{horizon}']

        self.scaler_tech = MinMaxScaler()
        self.scaler_macro = MinMaxScaler()

    def load_and_split(self, for_training=True) -> Tuple[Dict, Dict, Dict, Dict]:
        # ... (Phần code load dữ liệu giữ nguyên như cũ) ...
        self.logger.info(f"Đang đọc dữ liệu từ: {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {self.data_path}")
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)

        if for_training:
            df = df.dropna(subset=self.target_cols)

        data_tech_scaled = self.scaler_tech.fit_transform(df[self.tech_cols])
        data_macro_scaled = self.scaler_macro.fit_transform(df[self.macro_cols])
        targets = df[self.target_cols].values

        X_tech, X_macro, y = [], [], []
        for i in range(self.window_size, len(df)):
            tech_window = data_tech_scaled[i - self.window_size:i]
            macro_current = data_macro_scaled[i - 1]
            target_current = targets[i]
            X_tech.append(tech_window)
            X_macro.append(macro_current)
            y.append(target_current)

        X_tech = np.array(X_tech)
        X_macro = np.array(X_macro)
        y = np.array(y)

        split_idx = int(len(X_tech) * (1 - self.test_ratio))
        X_train = {'input_price': X_tech[:split_idx], 'input_macro': X_macro[:split_idx]}
        y_train = {'output_min': y[:split_idx, 0], 'output_max': y[:split_idx, 1]}
        X_test = {'input_price': X_tech[split_idx:], 'input_macro': X_macro[split_idx:]}
        y_test = {'output_min': y[split_idx:, 0], 'output_max': y[split_idx:, 1]}

        return X_train, y_train, X_test, y_test

    # --- SỬA HÀM NÀY: Thêm tham số suffix ---
    def save_scalers(self, suffix=""):
        """Lưu scaler với tên riêng biệt (VD: scaler_tech_short_term.pkl)"""
        os.makedirs(self.model_save_path, exist_ok=True)

        name_tech = f"scaler_tech{suffix}.pkl"
        name_macro = f"scaler_macro{suffix}.pkl"

        joblib.dump(self.scaler_tech, os.path.join(self.model_save_path, name_tech))
        joblib.dump(self.scaler_macro, os.path.join(self.model_save_path, name_macro))
        self.logger.info(f"💾 Đã lưu Scalers: {name_tech}, {name_macro}")
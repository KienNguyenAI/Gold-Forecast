import os
import logging
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict


class GoldPredictor:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.settings = settings

        # Đường dẫn
        self.model_path = os.path.join(settings['paths']['model_save'], f"{settings['model']['name']}_best.keras")
        self.scaler_path = settings['paths']['model_save']
        self.data_path = os.path.join(settings['paths']['processed_data'], "gold_processed_features.csv")

        # Load Model & Scalers
        self._load_artifacts()

    def _load_artifacts(self):
        """Tải Model và Scaler từ ổ cứng"""
        self.logger.info("📥 Đang tải Model và Scalers...")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Chưa tìm thấy Model tại {self.model_path}. Hãy chạy lệnh 'train' trước!")

        # Load Model
        self.model = tf.keras.models.load_model(self.model_path)

        # Load Scalers
        try:
            self.scaler_tech = joblib.load(os.path.join(self.scaler_path, "scaler_tech.pkl"))
            self.scaler_macro = joblib.load(os.path.join(self.scaler_path, "scaler_macro.pkl"))
        except FileNotFoundError:
            raise FileNotFoundError("❌ Thiếu file Scaler (.pkl). Hãy chạy lệnh 'train' trước!")

    def prepare_last_window(self):
        """Lấy dữ liệu 60 ngày cuối cùng để dự đoán ngày tiếp theo"""
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)

        window_size = self.settings['processing']['window_size']

        # Định nghĩa cột (Phải khớp với lúc train)
        tech_cols = ['Gold_Close', 'Log_Return', 'RSI', 'Volatility_20d', 'Trend_Signal']
        macro_cols = ['DXY', 'US10Y', 'CPI', 'Real_Rate']

        # Lấy 60 dòng cuối cùng
        last_window_df = df.tail(window_size)

        if len(last_window_df) < window_size:
            raise ValueError(f"Dữ liệu không đủ {window_size} ngày để dự đoán.")

        # Lấy giá đóng cửa ngày cuối cùng (để tính giá đích danh)
        current_price = last_window_df['Gold_Close'].iloc[-1]
        last_date = last_window_df.index[-1]

        # Scale dữ liệu
        tech_scaled = self.scaler_tech.transform(last_window_df[tech_cols])

        # Với Macro, ta lấy dòng cuối cùng (giả định vĩ mô ngày mai tương tự hôm nay)
        macro_last_row = last_window_df[macro_cols].iloc[[-1]]
        macro_scaled = self.scaler_macro.transform(macro_last_row)

        # Reshape cho đúng input của LSTM
        # Input Price: (1, 60, 5)
        # Input Macro: (1, 4)
        input_price = np.expand_dims(tech_scaled, axis=0)
        input_macro = macro_scaled  # Đã là (1, 4)

        return input_price, input_macro, current_price, last_date

    def predict(self):
        """Thực hiện dự đoán"""
        self.logger.info("🔮 Đang thực hiện dự đoán...")

        # 1. Chuẩn bị data
        X_price, X_macro, current_price, last_date = self.prepare_last_window()

        # 2. Predict
        # Model trả về list 2 phần tử: [pred_min, pred_max]
        predictions = self.model.predict([X_price, X_macro], verbose=0)

        pred_min_change = predictions[0][0][0]  # Output 1
        pred_max_change = predictions[1][0][0]  # Output 2

        # 3. Quy đổi từ % sang Giá USD
        price_min = current_price * (1 + pred_min_change)
        price_max = current_price * (1 + pred_max_change)

        # Logic đơn giản: Giá đóng cửa dự kiến (Trung bình Min/Max)
        price_close_forecast = (price_min + price_max) / 2

        result = {
            "last_date": last_date.strftime('%Y-%m-%d'),
            "current_price": current_price,
            "forecast_min": price_min,
            "forecast_max": price_max,
            "forecast_close": price_close_forecast,
            "change_pct_min": pred_min_change * 100,
            "change_pct_max": pred_max_change * 100
        }

        self._print_result(result)
        return result

    def _print_result(self, res):
        """In kết quả đẹp mắt"""
        print("\n" + "=" * 40)
        print(f"🌟 KẾT QUẢ DỰ BÁO GIÁ VÀNG")
        print("=" * 40)
        print(f"📅 Dựa trên dữ liệu đến ngày: {res['last_date']}")
        print(f"💰 Giá hiện tại:           ${res['current_price']:.2f}")
        print("-" * 40)
        print(f"📉 Giá Thấp nhất dự kiến:  ${res['forecast_min']:.2f} ({res['change_pct_min']:.2f}%)")
        print(f"📈 Giá Cao nhất dự kiến:   ${res['forecast_max']:.2f} ({res['change_pct_max']:.2f}%)")
        print("-" * 40)

        trend = "TĂNG 🟢" if res['forecast_close'] > res['current_price'] else "GIẢM 🔴"
        print(f"🎯 Xu hướng tổng thể:      {trend}")
        print("=" * 40 + "\n")
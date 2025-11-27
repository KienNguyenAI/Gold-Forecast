import os
import logging
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict
from datetime import datetime, timedelta
from .forecast_generator import ForecastGenerator

class GoldPredictor:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.settings = settings

        self.model_path = os.path.join(settings['paths']['model_save'], f"{settings['model']['name']}_best.keras")
        self.scaler_path = settings['paths']['model_save']
        self.processed_dir = settings['paths']['processed_data']
        self.final_dir = settings['paths'].get('final_data', 'data/final/')

        self.data_path = os.path.join(settings['paths']['processed_data'], "gold_processed_features.csv")

        self._load_artifacts()

    def _load_artifacts(self):
        self.logger.info("Đang tải Model và Scalers...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Chưa tìm thấy Model tại {self.model_path}")

        self.model = tf.keras.models.load_model(self.model_path)
        try:
            self.scaler_tech = joblib.load(os.path.join(self.scaler_path, "scaler_tech.pkl"))
            self.scaler_macro = joblib.load(os.path.join(self.scaler_path, "scaler_macro.pkl"))
        except FileNotFoundError:
            raise FileNotFoundError("Thiếu file Scaler.")

    def prepare_last_window(self):
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        window_size = self.settings['processing'].get('window_size', 30)  # Lấy từ config, mặc định 30

        # Định nghĩa cột (Phải khớp với lúc train)
        tech_cols = ['Gold_Close', 'Log_Return', 'RSI', 'Volatility_20d', 'Trend_Signal']
        macro_cols = ['DXY', 'US10Y', 'CPI', 'Real_Rate']  # Đảm bảo tên cột khớp với file processed

        last_window_df = df.tail(window_size)

        # Cho phép chạy dự báo ngay cả khi thiếu vài dòng (fallback)
        if len(last_window_df) < window_size:
            self.logger.warning(f"Dữ liệu hơi ít ({len(last_window_df)} dòng), kết quả có thể kém chính xác.")

        current_price = last_window_df['Gold_Close'].iloc[-1]
        last_date = last_window_df.index[-1]

        tech_scaled = self.scaler_tech.transform(last_window_df[tech_cols])
        macro_last_row = last_window_df[macro_cols].iloc[[-1]]
        macro_scaled = self.scaler_macro.transform(macro_last_row)

        input_price = np.expand_dims(tech_scaled, axis=0)
        input_macro = macro_scaled

        return input_price, input_macro, current_price, last_date

    def predict(self):
        self.logger.info("Đang thực hiện dự đoán...")

        X_price, X_macro, current_price, last_date = self.prepare_last_window()

        predictions = self.model.predict([X_price, X_macro], verbose=0)

        pred_min_change = predictions[0][0][0]
        pred_max_change = predictions[1][0][0]

        price_min = current_price * (1 + pred_min_change)
        price_max = current_price * (1 + pred_max_change)
        price_close_forecast = (price_min + price_max) / 2

        # --- TÍNH TOÁN NGÀY KẾT THÚC (FIX LỖI) ---
        prediction_days = self.settings['processing'].get('window_size', 30)

        end_date = last_date + timedelta(days=prediction_days)

        result = {
            "last_date": last_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "days": prediction_days,
            "current_price": current_price,
            "forecast_min": price_min,
            "forecast_max": price_max,
            "forecast_close": price_close_forecast,
            "change_pct_min": pred_min_change * 100,
            "change_pct_max": pred_max_change * 100
        }

        os.makedirs(self.final_dir, exist_ok=True)
        save_path = os.path.join(self.final_dir, f"{result['days']}day_forecast.csv")
        df_result = pd.DataFrame([result])
        df_result.to_csv(save_path, index=False)
        self.logger.info(f"Đã lưu kết quả dự báo FINAL vào: {save_path}")

        generator = ForecastGenerator()
        generator.generate(result, self.final_dir)

        self._print_result(result)
        return result

    def _print_result(self, res):
        print("\n" + "=" * 50)
        print(f"KẾT QUẢ DỰ BÁO GIÁ VÀNG ({res['days']} NGÀY TỚI)")
        print("=" * 50)
        print(f"Dữ liệu đến ngày:      {res['last_date']}")
        print(f"Dự báo đến ngày:      {res['end_date']}")
        print(f"Giá hiện tại:          ${res['current_price']:.2f}")
        print("-" * 50)
        print(f"Đáy dự kiến:           ${res['forecast_min']:.2f} ({res['change_pct_min']:.2f}%)")
        print(f"Đỉnh dự kiến:          ${res['forecast_max']:.2f} ({res['change_pct_max']:.2f}%)")
        print("-" * 50)

        avg = (res['forecast_min'] + res['forecast_max']) / 2
        trend = "TĂNG" if avg > res['current_price'] else "GIẢM"
        print(f"Xu hướng tổng thể:      {trend}")
        print("=" * 50 + "\n")
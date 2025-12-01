import os
import logging
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict
from datetime import datetime, timedelta


class GoldPredictor:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.settings = settings

        # Đường dẫn model & data
        self.model_path = os.path.join(settings['paths']['model_save'], f"{settings['model']['name']}_best.keras")
        self.scaler_path = settings['paths']['model_save']
        self.data_path = os.path.join(settings['paths']['processed_data'], "gold_processed_features.csv")
        self.final_dir = settings['paths'].get('final_data', 'data/final/')

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

    def _get_input_for_date(self, full_df, lookback_date):
        """
        Tìm dòng dữ liệu gần nhất với 'lookback_date' để làm đầu vào cho model.
        """
        window_size = self.settings['processing'].get('window_size', 60)

        # Tìm index của ngày gần nhất (<= lookback_date)
        # asof: Tìm giá trị index gần nhất phía trước
        try:
            loc_idx = full_df.index.get_indexer([lookback_date], method='pad')[0]
        except:
            return None, None, None, None

        if loc_idx < window_size:
            return None, None, None, None

        # Cắt window 60 ngày kết thúc tại loc_idx
        # loc_idx là vị trí trong mảng (integer), ta lấy từ (loc_idx - window + 1) đến (loc_idx + 1)
        sub_df = full_df.iloc[loc_idx - window_size + 1: loc_idx + 1]

        # Kiểm tra lại xem ngày cuối cùng của sub_df có quá xa lookback_date không?
        # Nếu data bị lủng lỗ quá 5 ngày thì bỏ qua để tránh sai số
        actual_date = sub_df.index[-1]
        if (lookback_date - actual_date).days > 5:
            return None, None, None, None

        # --- Chuẩn bị dữ liệu ---
        tech_cols = ['Gold_Close', 'Log_Return', 'RSI', 'Volatility_20d', 'Trend_Signal']
        macro_cols = ['DXY', 'US10Y', 'CPI', 'Real_Rate']

        ref_price = sub_df['Gold_Close'].iloc[-1]

        tech_scaled = self.scaler_tech.transform(sub_df[tech_cols])
        macro_last_row = sub_df[macro_cols].iloc[[-1]]
        macro_scaled = self.scaler_macro.transform(macro_last_row)

        input_price = np.expand_dims(tech_scaled, axis=0)
        input_macro = macro_scaled

        return input_price, input_macro, ref_price, actual_date

    def predict(self):
        self.logger.info("🚀 Đang thực hiện dự đoán Tương lai (Future Forecast)...")

        # 1. Load dữ liệu
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        df = df.sort_index()  # Đảm bảo sort theo ngày

        last_history_date = df.index[-1]
        forecast_horizon = self.settings['processing'].get('forecast_horizon', 30)

        forecast_results = []

        self.logger.info(f"Ngày dữ liệu cuối cùng: {last_history_date.date()}")
        self.logger.info(f"Bắt đầu dự báo cho: {last_history_date.date() + timedelta(days=1)} trở đi.")

        # 2. VÒNG LẶP DỰ BÁO THEO NGÀY TƯƠNG LAI
        # Chạy từ T+1 đến T+30
        for i in range(1, forecast_horizon + 1):
            # Ngày Đích (Target) muốn dự báo
            target_date = last_history_date + timedelta(days=i)

            # Ngày Cần (Input): Để dự báo cho Target, ta cần dữ liệu của 30 ngày trước đó
            lookback_date = target_date - timedelta(days=forecast_horizon)

            # Tìm dữ liệu input tương ứng với lookback_date
            X_price, X_macro, ref_price, actual_input_date = self._get_input_for_date(df, lookback_date)

            if X_price is not None:
                # AI DỰ ĐOÁN
                pred = self.model.predict([X_price, X_macro], verbose=0)

                pred_min_pct = pred[0][0][0]
                pred_max_pct = pred[1][0][0]

                # Tính giá (Dựa trên giá của ngày lookback)
                forecast_min = ref_price * (1 + pred_min_pct)
                forecast_max = ref_price * (1 + pred_max_pct)
                forecast_close = (forecast_min + forecast_max) / 2

                forecast_results.append({
                    'Date': target_date,
                    'Forecast_Close': forecast_close,
                    'Forecast_Min': forecast_min,
                    'Forecast_Max': forecast_max
                })
            else:
                # Nếu không tìm thấy input (ví dụ lookback rơi vào ngày nghỉ quá xa),
                # ta có thể skip hoặc fill bằng dữ liệu ngày hôm trước (forward fill logic)
                # Ở đây ta skip để an toàn
                pass

        # 3. Lưu kết quả
        df_forecast = pd.DataFrame(forecast_results)

        if df_forecast.empty:
            self.logger.error("❌ Không sinh được dữ liệu dự báo nào!")
            return None

        os.makedirs(self.final_dir, exist_ok=True)
        save_path = os.path.join(self.final_dir, "30days_forecast.csv")
        df_forecast.to_csv(save_path, index=False)

        self.logger.info(f"✅ Đã lưu dự báo: {save_path}")

        # Print info
        first_date = df_forecast.iloc[0]['Date'].strftime('%Y-%m-%d')
        last_date = df_forecast.iloc[-1]['Date'].strftime('%Y-%m-%d')
        print("\n" + "=" * 50)
        print(f"KHOẢNG THỜI GIAN DỰ BÁO: {first_date} -> {last_date}")
        print("=" * 50 + "\n")

        return {
            "last_date": last_history_date.strftime('%Y-%m-%d'),
            "end_date": last_date,
            "days": len(df_forecast),
            "current_price": df['Gold_Close'].iloc[-1],
            "forecast_close": df_forecast.iloc[-1]['Forecast_Close'],
            "forecast_min": df_forecast.iloc[-1]['Forecast_Min'],
            "forecast_max": df_forecast.iloc[-1]['Forecast_Max']
        }
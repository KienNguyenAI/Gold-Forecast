import pandas as pd
import numpy as np
import os
import logging
from datetime import timedelta


class ForecastGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate(self, prediction_result: dict, output_dir: str):
        """
        Tạo dữ liệu chi tiết từng ngày từ kết quả dự báo tóm tắt.
        Lưu thành file: {days}days_forecast.csv
        """
        self.logger.info("🎲 Đang sinh dữ liệu chi tiết (Dummy Path)...")

        try:
            # 1. Lấy thông tin từ kết quả dự báo
            last_date = pd.to_datetime(prediction_result['last_date'])
            end_date = pd.to_datetime(prediction_result['end_date'])
            days_count = int(prediction_result['days'])

            current_price = prediction_result['current_price']
            target_close = prediction_result['forecast_close']
            target_min = prediction_result['forecast_min']
            target_max = prediction_result['forecast_max']

            # 2. Tạo danh sách ngày (bắt đầu từ ngày mai)
            start_gen_date = last_date + timedelta(days=1)
            dates = pd.date_range(start=start_gen_date, end=end_date, freq='D')

            # Kiểm tra độ dài (đôi khi date_range có thể lệch 1 ngày do giờ giấc)
            if len(dates) != days_count:
                # Fallback: ép đúng số ngày
                dates = pd.date_range(start=start_gen_date, periods=days_count, freq='D')

            # 3. Sinh dữ liệu giá (Linear Interpolation + Noise)
            # Tạo đường xu hướng tuyến tính từ Giá hiện tại -> Giá dự báo
            trend_line = np.linspace(current_price, target_close, days_count)

            # Thêm nhiễu (Random Noise) để nhìn giống thật
            # Giả định biến động 0.5% mỗi ngày
            np.random.seed(42)  # Cố định seed để kết quả nhất quán mỗi lần chạy
            noise = np.random.normal(0, current_price * 0.01, days_count)
            generated_prices = trend_line + noise

            # 4. Sinh dữ liệu dải Min/Max (Hình nón mở rộng dần)
            # Min: Đi từ Current -> Target Min
            lower_bound = np.linspace(current_price, target_min, days_count)
            # Max: Đi từ Current -> Target Max
            upper_bound = np.linspace(current_price, target_max, days_count)

            # (Optional) Clip giá nằm trong dải Min/Max để logic không bị vỡ
            # generated_prices = np.clip(generated_prices, lower_bound, upper_bound)

            # 5. Tạo DataFrame
            df_detail = pd.DataFrame({
                'Date': dates,
                'Forecast_Close': generated_prices,
                'Forecast_Min': lower_bound,
                'Forecast_Max': upper_bound
            })

            # 6. Lưu file CSV
            filename = f"{days_count}days_forecast.csv"
            save_path = os.path.join(output_dir, filename)

            os.makedirs(output_dir, exist_ok=True)
            df_detail.to_csv(save_path, index=False)

            self.logger.info(f"✅ Đã tạo dữ liệu chi tiết: {save_path}")
            return save_path

        except Exception as e:
            self.logger.error(f"❌ Lỗi khi sinh dữ liệu chi tiết: {e}")
            return None
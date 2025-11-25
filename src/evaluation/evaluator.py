import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict
from src.training.data_provider import DataProvider


class ModelEvaluator:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.provider = DataProvider(settings)

        model_name = settings['model']['name']
        self.model_path = os.path.join(settings['paths']['model_save'], f"{model_name}_best.keras")

    def run(self):
        self.logger.info("📊 Đang tính toán các chỉ số hiệu suất...")

        # 1. Load dữ liệu Test
        # Lưu ý: for_training=True để đảm bảo có Target để so sánh
        _, _, X_test, y_test = self.provider.load_and_split(for_training=True)

        # 2. Load Model & Predict
        if not os.path.exists(self.model_path):
            self.logger.error("❌ Chưa có model để đánh giá.")
            return

        model = tf.keras.models.load_model(self.model_path)
        preds = model.predict([X_test['input_price'], X_test['input_macro']], verbose=0)

        # Tách dự báo Min/Max
        pred_min = preds[0].flatten()
        pred_max = preds[1].flatten()
        pred_avg = (pred_min + pred_max) / 2  # Giá dự báo trung bình

        # Lấy thực tế
        actual_min = y_test['output_min']
        actual_max = y_test['output_max']
        actual_avg = (actual_min + actual_max) / 2

        # --- TÍNH TOÁN CHỈ SỐ ---
        self._calculate_metrics(actual_avg, pred_avg, pred_min, pred_max)

    def _calculate_metrics(self, actual, predicted, pred_min, pred_max):
        # 1. Regression Metrics (Độ sai số)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)

        # MAPE (Mean Absolute Percentage Error)
        # Tránh chia cho 0 bằng cách cộng epsilon nhỏ
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100

        # 2. Direction Accuracy (Độ chính xác xu hướng)
        # Nếu cùng dấu (cùng dương hoặc cùng âm) -> Đoán đúng hướng
        correct_direction = np.sign(predicted) == np.sign(actual)
        direction_acc = np.mean(correct_direction) * 100

        # 3. Risk Metrics (Giả lập giao dịch trên tập Test)
        # Giả sử: Mua nếu dự báo > 0, Bán nếu dự báo < 0
        signals = np.sign(predicted)
        returns = actual * signals  # Lợi nhuận từng ngày
        cumulative_returns = np.cumsum(returns)

        # Max Drawdown (Mức sụt giảm tài khoản lớn nhất)
        peak = np.maximum.accumulate(cumulative_returns)
        # Tránh chia cho 0 nếu peak = 0
        drawdown = (cumulative_returns - peak) / (np.abs(peak) + 1e-8)
        max_drawdown = np.min(drawdown) * 100

        # Sharpe Ratio (Hiệu suất/Rủi ro) - Giả định lãi suất phi rủi ro = 0
        # Nhân căn(252) để quy đổi ra năm
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)

        # Win Rate
        win_rate = np.mean(returns > 0) * 100

        # --- IN BÁO CÁO ---
        print("\n" + "=" * 50)
        print("📊 BÁO CÁO HIỆU SUẤT MÔ HÌNH (MODEL EVALUATION)")
        print("=" * 50)
        print("1. ĐỘ CHÍNH XÁC DỰ BÁO (REGRESSION):")
        print(f"   - MAE (Sai số tuyệt đối):   {mae:.4f} ")
        print(f"   - RMSE (Sai số bình phương):{rmse:.4f}")
        print(f"   - MAPE (Sai số phần trăm):  {mape:.2f}%")
        print(f"   - R² Score (Độ phù hợp):    {r2:.4f} ")

        print("\n2. ĐỘ CHÍNH XÁC XU HƯỚNG (DIRECTION):")
        print(f"   - Accuracy (Đoán đúng Tăng/Giảm): {direction_acc:.2f}%")

        print("\n3. CHỈ SỐ TÀI CHÍNH (RISK & STRATEGY):")
        print(f"   - Win Rate (Tỷ lệ thắng lệnh):    {win_rate:.2f}%")
        print(f"   - Max Drawdown (Rủi ro sụt giảm): {max_drawdown:.2f}%")
        print(f"   - Sharpe Ratio (Hiệu quả đầu tư): {sharpe:.2f} (>1 là tốt)")
        print("=" * 50 + "\n")
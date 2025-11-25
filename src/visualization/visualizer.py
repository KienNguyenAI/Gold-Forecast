import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import logging
from typing import Dict
from src.training.data_provider import DataProvider
from src.prediction import GoldPredictor
import random

class Visualizer:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.provider = DataProvider(settings)

        model_name = settings['model']['name']
        self.model_path = os.path.join(settings['paths']['model_save'], f"{model_name}_best.keras")
        self.figures_dir = settings['paths']['figures_save']

    def plot_forecast(self, days_to_plot=100):
        """Vẽ biểu đồ Dự báo tương lai (Code cũ giữ nguyên)"""
        self.logger.info("🎨 Đang vẽ biểu đồ dự báo kết quả...")
        df = pd.read_csv(self.provider.data_path, index_col=0, parse_dates=True)
        recent_df = df.tail(days_to_plot)
        dates = recent_df.index
        prices = recent_df['Gold_Close']
        current_date = dates[-1]
        current_price = prices.iloc[-1]

        try:
            predictor = GoldPredictor(self.settings)
            res = predictor.predict()
            end_date = pd.Timestamp(res['end_date'])
            forecast_min = res['forecast_min']
            forecast_max = res['forecast_max']
        except Exception as e:
            self.logger.error(f"❌ Không thể lấy dự báo: {e}")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(dates, prices, label='Lịch sử giá', color='black', linewidth=1.5)
        plt.scatter([current_date], [current_price], color='blue', zorder=5, label='Hiện tại')
        plt.text(current_date, current_price, f" ${current_price:.0f}", verticalalignment='bottom', fontsize=9)

        plt.plot([current_date, end_date], [current_price, forecast_min], color='red', linestyle='--', alpha=0.5)
        plt.plot([current_date, end_date], [current_price, forecast_max], color='green', linestyle='--', alpha=0.5)
        plt.fill_between([current_date, end_date], [current_price, forecast_min], [current_price, forecast_max],
                         color='green', alpha=0.1, label='Vùng dự báo AI')

        plt.scatter([end_date], [forecast_min], color='red', marker='v', zorder=5)
        plt.text(end_date, forecast_min, f" Min: ${forecast_min:.0f}", color='red', verticalalignment='top')
        plt.scatter([end_date], [forecast_max], color='green', marker='^', zorder=5)
        plt.text(end_date, forecast_max, f" Max: ${forecast_max:.0f}", color='green', verticalalignment='bottom')

        plt.title(f"Dự báo Giá Vàng AI (Từ {current_date.strftime('%Y-%m-%d')} đến {end_date.strftime('%Y-%m-%d')})")
        plt.xlabel("Thời gian")
        plt.ylabel("Giá Vàng (USD)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        os.makedirs(self.figures_dir, exist_ok=True)
        save_path = os.path.join(self.figures_dir, "forecast_result_final.png")
        plt.savefig(save_path)
        self.logger.info(f"✅ Đã lưu biểu đồ dự báo tại: {save_path}")

    def plot_test_results(self):
        """
        👇 [MỚI] Hàm vẽ biểu đồ so sánh Thực tế vs Dự báo trên tập Test
        """
        self.logger.info("📊 Đang vẽ biểu đồ kiểm định trên tập Test...")

        # 1. Lấy dữ liệu Test (for_training=True để lấy đúng target)
        _, _, X_test, y_test = self.provider.load_and_split(for_training=True)

        # 2. Load Model & Predict
        if not os.path.exists(self.model_path):
            self.logger.error("❌ Chưa có model.")
            return

        model = tf.keras.models.load_model(self.model_path)
        preds = model.predict([X_test['input_price'], X_test['input_macro']], verbose=0)

        # Tách output
        pred_min = preds[0].flatten()
        pred_max = preds[1].flatten()

        actual_min = y_test['output_min']
        actual_max = y_test['output_max']

        # 3. Lấy ngày tháng tương ứng
        # (Thủ thuật: Lấy n ngày cuối cùng của file dữ liệu gốc, với n = số lượng mẫu test)
        df = pd.read_csv(self.provider.data_path, index_col=0, parse_dates=True)
        # Lọc dòng thiếu target trước khi lấy index (để khớp với logic của load_and_split)
        df_clean = df.dropna(subset=self.provider.target_cols)
        test_dates = df_clean.index[-len(actual_min):]

        # 4. Vẽ biểu đồ so sánh
        plt.figure(figsize=(14, 8))

        # Subplot 1: Min Change
        plt.subplot(2, 1, 1)
        plt.plot(test_dates, actual_min, label='Thực tế (Min)', color='gray', alpha=0.7, linewidth=1)
        plt.plot(test_dates, pred_min, label='AI Dự báo (Min)', color='red', alpha=0.8, linewidth=1.5, linestyle='--')
        plt.title('Kiểm định: Biến động giá THẤP NHẤT (Min % Change)')
        plt.ylabel('% Thay đổi')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Max Change
        plt.subplot(2, 1, 2)
        plt.plot(test_dates, actual_max, label='Thực tế (Max)', color='gray', alpha=0.7, linewidth=1)
        plt.plot(test_dates, pred_max, label='AI Dự báo (Max)', color='green', alpha=0.8, linewidth=1.5, linestyle='--')
        plt.title('Kiểm định: Biến động giá CAO NHẤT (Max % Change)')
        plt.ylabel('% Thay đổi')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Lưu ảnh
        os.makedirs(self.figures_dir, exist_ok=True)
        save_path = os.path.join(self.figures_dir, "test_evaluation_chart.png")
        plt.savefig(save_path)
        self.logger.info(f"✅ Đã lưu biểu đồ kiểm định tại: {save_path}")

    def plot_test_simulation(self):
        """
        🔍 Kiểm chứng quá khứ: Chọn 1 ngày ngẫu nhiên trong tập Test,
        vẽ vùng dự báo và so sánh với giá chạy thực tế.
        """
        self.logger.info("🎲 Đang chạy mô phỏng kiểm chứng trên tập Test...")

        # 1. Load Data & Model
        # for_training=False để lấy full dữ liệu
        df = pd.read_csv(self.provider.data_path, index_col=0, parse_dates=True)

        if not os.path.exists(self.model_path):
            self.logger.error("❌ Chưa có model.")
            return
        model = tf.keras.models.load_model(self.model_path)

        # Load Scaler từ file (để scale input thủ công)
        import joblib
        scaler_path = self.settings['paths']['model_save']
        scaler_tech = joblib.load(os.path.join(scaler_path, "scaler_tech.pkl"))
        scaler_macro = joblib.load(os.path.join(scaler_path, "scaler_macro.pkl"))

        # 2. Chọn ngẫu nhiên 1 điểm trong quá khứ
        window_size = self.settings['processing']['window_size']
        prediction_days = 30  # Mặc định 30 ngày

        # Chỉ chọn điểm nào có đủ 30 ngày tương lai để so sánh
        valid_range = len(df) - prediction_days
        if valid_range <= window_size:
            self.logger.warning("Dữ liệu quá ngắn để mô phỏng.")
            return

        # Chọn random index (đảm bảo nằm trong tập Test - 20% cuối)
        test_start_idx = int(len(df) * 0.8)
        random_idx = random.randint(test_start_idx, valid_range - 1)

        # 3. Lấy dữ liệu tại điểm đó (Giả lập quá khứ)
        # Input (60 ngày trước điểm đó)
        input_df = df.iloc[random_idx - window_size: random_idx]

        # Ground Truth (30 ngày sau điểm đó)
        future_df = df.iloc[random_idx: random_idx + prediction_days]

        # Thông tin điểm "Hiện tại" (trong quá khứ)
        current_date = input_df.index[-1]
        current_price = input_df['Gold_Close'].iloc[-1]
        end_date = future_df.index[-1]

        # 4. Chuẩn bị Input cho Model
        tech_cols = ['Gold_Close', 'Log_Return', 'RSI', 'Volatility_20d', 'Trend_Signal']
        macro_cols = ['DXY', 'US10Y', 'CPI', 'Real_Rate']

        tech_scaled = scaler_tech.transform(input_df[tech_cols])
        macro_last = input_df[macro_cols].iloc[[-1]]
        macro_scaled = scaler_macro.transform(macro_last)

        X_price = np.expand_dims(tech_scaled, axis=0)
        X_macro = macro_scaled

        # 5. Dự báo
        preds = model.predict([X_price, X_macro], verbose=0)
        pred_min_pct = preds[0][0][0]
        pred_max_pct = preds[1][0][0]

        # Quy đổi ra giá
        forecast_min = current_price * (1 + pred_min_pct)
        forecast_max = current_price * (1 + pred_max_pct)

        # 6. Vẽ Biểu Đồ (Matplotlib)
        plt.figure(figsize=(14, 7))

        # A. Vẽ quá khứ (60 ngày)
        plt.plot(input_df.index, input_df['Gold_Close'], color='black', label='Lịch sử (Input)')

        # B. Vẽ tương lai THỰC TẾ (30 ngày) - Đường màu xanh dương đậm
        plt.plot(future_df.index, future_df['Gold_Close'], color='blue', linewidth=2, label='Giá chạy thực tế (Actual)')

        # C. Vẽ điểm hiện tại
        plt.scatter([current_date], [current_price], color='blue', s=100, zorder=5)
        plt.text(current_date, current_price, f" Start: ${current_price:.0f}", verticalalignment='bottom')

        # D. Vẽ Vùng Dự Báo AI (Tam giác xanh nhạt)
        plt.plot([current_date, end_date], [current_price, forecast_min], color='red', linestyle='--', alpha=0.5)
        plt.plot([current_date, end_date], [current_price, forecast_max], color='green', linestyle='--', alpha=0.5)
        plt.fill_between([current_date, end_date],
                         [current_price, forecast_min],
                         [current_price, forecast_max],
                         color='green', alpha=0.15, label='Vùng dự báo AI')

        # E. Đánh dấu Min/Max Dự báo
        plt.scatter([end_date], [forecast_min], color='red', marker='v', s=80)
        plt.text(end_date, forecast_min, f" AI Min: ${forecast_min:.0f}", color='red', verticalalignment='top')

        plt.scatter([end_date], [forecast_max], color='green', marker='^', s=80)
        plt.text(end_date, forecast_max, f" AI Max: ${forecast_max:.0f}", color='green', verticalalignment='bottom')

        # Trang trí
        plt.title(f"Kiểm chứng Dự báo AI (Ngày mô phỏng: {current_date.strftime('%Y-%m-%d')})")
        plt.xlabel("Thời gian")
        plt.ylabel("Giá Vàng")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 7. Lưu ảnh
        os.makedirs(self.figures_dir, exist_ok=True)
        save_path = os.path.join(self.figures_dir, "test_simulation_case.png")
        plt.savefig(save_path)
        self.logger.info(f"✅ Đã lưu biểu đồ mô phỏng tại: {save_path}")
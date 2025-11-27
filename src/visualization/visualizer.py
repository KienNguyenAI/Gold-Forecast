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
from datetime import datetime

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
        📊 Vẽ biểu đồ so sánh YTD (Từ đầu năm đến nay)
        Gộp cả Min/Max vào chung 1 biểu đồ để dễ nhìn.
        """
        self.logger.info("📊 Đang vẽ biểu đồ kiểm định YTD (Year-To-Date)...")

        # 1. Load Data & Model
        _, _, X_test, y_test = self.provider.load_and_split(for_training=True)

        if not os.path.exists(self.model_path):
            self.logger.error("❌ Chưa có model.")
            return

        model = tf.keras.models.load_model(self.model_path)
        preds = model.predict([X_test['input_price'], X_test['input_macro']], verbose=0)

        # 2. Chuẩn bị dữ liệu % Change
        pred_min_pct = preds[0].flatten()
        pred_max_pct = preds[1].flatten()

        # 3. Lấy dữ liệu gốc để quy đổi ra Giá ($)
        df = pd.read_csv(self.provider.data_path, index_col=0, parse_dates=True)
        df_clean = df.dropna(subset=self.provider.target_cols)

        # Lấy đoạn dữ liệu tương ứng với tập Test
        test_slice = df_clean.iloc[-len(pred_min_pct):]
        test_dates = test_slice.index
        base_prices = test_slice['Gold_Close'].values

        # 4. Quy đổi ra Giá USD
        pred_price_min = base_prices * (1 + pred_min_pct)
        pred_price_max = base_prices * (1 + pred_max_pct)

        # Giá thực tế (dùng giá Close làm tham chiếu chính)
        actual_prices = base_prices

        # 5. LỌC DỮ LIỆU YTD (CHỈ LẤY TỪ ĐẦU NĂM NAY)
        current_year = datetime.now().year
        # Hoặc nếu data của bạn ở tương lai (2025), hãy lấy năm của data:
        # current_year = test_dates[-1].year

        # Tạo DataFrame tạm để lọc cho dễ
        eval_df = pd.DataFrame({
            'Date': test_dates,
            'Actual_Close': actual_prices,
            'AI_Min': pred_price_min,
            'AI_Max': pred_price_max
        })
        eval_df.set_index('Date', inplace=True)

        # Lọc lấy năm hiện tại (VD: 2025)
        ytd_df = eval_df[eval_df.index.year == current_year]

        if ytd_df.empty:
            self.logger.warning(f"⚠️ Không có dữ liệu test cho năm {current_year}. Vẽ toàn bộ test set.")
            ytd_df = eval_df  # Fallback nếu không có data năm nay

        # 6. VẼ BIỂU ĐỒ GỘP (COMBINED CHART)
        plt.figure(figsize=(15, 8))

        dates = ytd_df.index

        # A. Vẽ Vùng Dự Báo AI (Màu xanh lá nhạt)
        plt.fill_between(dates, ytd_df['AI_Min'], ytd_df['AI_Max'],
                         color='green', alpha=0.15, label='Vùng An Toàn AI (Risk Range)')

        # B. Vẽ biên Min/Max của AI (Nét đứt)
        plt.plot(dates, ytd_df['AI_Min'], color='green', linestyle=':', linewidth=1, alpha=0.6)
        plt.plot(dates, ytd_df['AI_Max'], color='green', linestyle=':', linewidth=1, alpha=0.6)

        # C. Vẽ Giá Thực Tế (Màu Đen/Xanh đậm)
        plt.plot(dates, ytd_df['Actual_Close'], color='#1f77b4', linewidth=2, label='Giá Thực Tế (Close)')

        # D. Đánh dấu những điểm giá vọt ra khỏi vùng dự báo (Outliers)
        # Để xem khi nào AI bị sai
        outliers = ytd_df[(ytd_df['Actual_Close'] < ytd_df['AI_Min']) | (ytd_df['Actual_Close'] > ytd_df['AI_Max'])]
        if not outliers.empty:
            plt.scatter(outliers.index, outliers['Actual_Close'], color='red', s=30, marker='x',
                        label='Ngoại lệ (AI Sai)')

        # Trang trí
        plt.title(f'Hiệu suất AI từ đầu năm {current_year} đến nay (YTD Evaluation)', fontsize=14)
        plt.ylabel('Giá Vàng (USD)')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

        # Lưu ảnh
        os.makedirs(self.figures_dir, exist_ok=True)
        save_path = os.path.join(self.figures_dir, "test_evaluation_chart.png")
        plt.savefig(save_path)
        self.logger.info(f"✅ Đã lưu biểu đồ YTD (USD) tại: {save_path}")

    def plot_test_simulation(self):
        """
        Kiểm chứng quá khứ: Chọn 1 ngày ngẫu nhiên trong tập Test,
        vẽ vùng dự báo và so sánh với giá chạy thực tế.
        """
        self.logger.info("Đang chạy mô phỏng kiểm chứng trên tập Test...")

        # 1. Load Data & Model
        # for_training=False để lấy full dữ liệu
        df = pd.read_csv(self.provider.data_path, index_col=0, parse_dates=True)

        if not os.path.exists(self.model_path):
            self.logger.error("Chưa có model.")
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
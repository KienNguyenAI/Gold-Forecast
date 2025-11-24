import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
import logging
from typing import Dict
from src.training.data_provider import DataProvider


class Visualizer:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.provider = DataProvider(settings)

        model_name = settings['model']['name']
        self.model_path = os.path.join(settings['paths']['model_save'], f"{model_name}_best.keras")
        self.figures_dir = settings['paths']['figures_save']

    def plot_forecast(self, days_to_plot=200):
        self.logger.info("🎨 Đang vẽ biểu đồ dự báo...")

        # 1. Load Data
        _, _, X_test, y_test = self.provider.load_and_split()

        # 2. Load Model
        if not os.path.exists(self.model_path):
            self.logger.error("❌ Chưa có model.")
            return
        model = tf.keras.models.load_model(self.model_path)

        # 3. Predict
        preds = model.predict([X_test['input_price'], X_test['input_macro']], verbose=0)

        # Lấy dữ liệu đoạn cuối để vẽ
        pred_min_pct = preds[0].flatten()[-days_to_plot:]
        pred_max_pct = preds[1].flatten()[-days_to_plot:]

        # Lấy giá thực tế
        df = pd.read_csv(self.provider.data_path, index_col=0, parse_dates=True)
        real_prices = df['Gold_Close'].iloc[-days_to_plot:].values
        dates = df.index[-days_to_plot:]

        # Tính vùng dự báo
        # Lưu ý: Đây là minh họa vùng dự báo nếu ta biết trước tương lai
        # Thực tế nên vẽ dự báo one-step-ahead (t+1)
        forecast_lower = real_prices * (1 + pred_min_pct)
        forecast_upper = real_prices * (1 + pred_max_pct)

        plt.figure(figsize=(15, 7))
        plt.plot(dates, real_prices, label='Real Price', color='black')
        plt.fill_between(dates, forecast_lower, forecast_upper, color='green', alpha=0.2, label='AI Forecast Range')

        plt.title(f'AI Vision (Last {days_to_plot} days)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.figures_dir, "forecast_vision.png")
        plt.savefig(save_path)
        self.logger.info(f"📉 Đã lưu biểu đồ Vision tại: {save_path}")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import logging
from typing import Dict
from src.training.data_provider import DataProvider


class Visualizer:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.provider = DataProvider(settings)

        # Trỏ thẳng vào cấu hình model Long Term (30 ngày)
        if 'long_term' in settings.get('models_config', {}):
            model_name = settings['models_config']['long_term']['name']
            # Đồng bộ window_size của DataProvider với model 30 ngày (90 ngày)
            self.settings['processing']['window_size'] = settings['models_config']['long_term']['window_size']
        else:
            model_name = settings['model']['name']

        self.model_path = os.path.join(settings['paths']['model_save'], f"{model_name}.keras")
        self.figures_dir = settings['paths']['figures_save']

    def plot_test_results(self):
        """
        📊 Vẽ biểu đồ kiểm định trên tập Test mô hình 30 ngày.
        Hiển thị giá thực tế nằm trong vùng dải dự báo AI (giống ảnh AI Vision).
        """
        self.logger.info("📊 Đang vẽ biểu đồ kiểm định tập Test mô hình 30 ngày...")

        # 1. Load Model và Dữ liệu Test
        if not os.path.exists(self.model_path):
            self.logger.error(f"❌ Không tìm thấy model tại {self.model_path}")
            return

        # Load data với window_size đã đồng bộ (90)
        _, _, X_test, _ = self.provider.load_and_split(for_training=True)
        model = tf.keras.models.load_model(self.model_path, compile=False)

        # Dự báo biên độ % trên tập test
        preds = model.predict([X_test['input_price'], X_test['input_macro']], verbose=0)
        pred_min_pct = preds[0].flatten()
        pred_max_pct = preds[1].flatten()

        # 2. Xử lý dữ liệu hiển thị
        df = pd.read_csv(self.provider.data_path, index_col=0, parse_dates=True)
        # Lấy phần đuôi dữ liệu tương ứng với kích thước tập Test
        test_slice = df.iloc[-len(pred_min_pct):]
        test_dates = test_slice.index
        actual_prices = test_slice['Gold_Close'].values

        # Tính toán giá trị USD cho dải biên độ
        ai_min_price = actual_prices * (1 + pred_min_pct)
        ai_max_price = actual_prices * (1 + pred_max_pct)

        # 3. Vẽ biểu đồ theo style AI Vision
        plt.figure(figsize=(16, 8))

        # Vẽ dải biên độ dự báo (Màu xanh nhạt)
        plt.fill_between(test_dates, ai_min_price, ai_max_price,
                         color='green', alpha=0.2, label='AI Forecast Range')

        # Vẽ đường biên (Dotted lines) để rõ ràng hơn
        plt.plot(test_dates, ai_min_price, color='green', linestyle=':', linewidth=0.5, alpha=0.3)
        plt.plot(test_dates, ai_max_price, color='green', linestyle=':', linewidth=0.5, alpha=0.3)

        # Vẽ giá thực tế (Đường màu đen)
        plt.plot(test_dates, actual_prices, color='black', linewidth=1.5, label='Real Price')

        # 4. Trang trí biểu đồ
        plt.title(f"AI Vision - 30D Model Test Validation (Last {len(test_dates)} days)", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Gold Price (USD)", fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3)

        # Lưu kết quả
        os.makedirs(self.figures_dir, exist_ok=True)
        save_path = os.path.join(self.figures_dir, "test_evaluation_30d.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        self.logger.info(f"✅ Đã lưu biểu đồ kiểm định tại: {save_path}")

    def plot_forecast(self):
        """Giữ phương thức này để main.py không lỗi, nhưng tập trung vào plot_test_results theo yêu cầu."""
        self.plot_test_results()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from training.data_provider import DataProvider


class Visualizer:
    def __init__(self):
        self.provider = DataProvider(window_size=60)
        self.model_path = "models/best_gold_model.keras"

    def plot_forecast(self, days_to_plot=200):
        """
        Vẽ biểu đồ giá thực tế kẹp giữa vùng dự báo Min/Max
        days_to_plot: Chỉ vẽ 200 ngày cuối cho dễ nhìn
        """
        print("🎨 Đang chuẩn bị dữ liệu để vẽ tranh...")

        # 1. Load Data & Model
        _, _, X_test, y_test = self.provider.load_and_split(train_ratio=0.8)
        model = tf.keras.models.load_model(self.model_path)

        # 2. Dự báo
        preds = model.predict(X_test, verbose=0)
        pred_min_pct = preds[0].flatten()[-days_to_plot:]
        pred_max_pct = preds[1].flatten()[-days_to_plot:]

        # 3. Lấy giá gốc để giải mã
        df = pd.read_csv(self.provider.data_path, index_col=0, parse_dates=True)
        # Lấy phần dữ liệu tương ứng với X_test cuối cùng
        real_prices = df['Gold_Close'].iloc[-days_to_plot:].values
        dates = df.index[-days_to_plot:]

        # 4. Tính toán vùng giá dự báo (Tuyệt đối)
        # Công thức: Dự báo Min = Giá hiện tại * (1 + %Min dự báo)
        # Lưu ý: pred_min_pct[i] là dự báo cho 30 ngày SAU ngày i.
        # Để vẽ đẹp, ta sẽ vẽ vùng mây bao quanh giá hiện tại dựa trên dự báo của quá khứ
        # Nhưng cách trực quan nhất là: Tại ngày hôm nay, AI bảo vùng giá tới là bao nhiêu?

        forecast_lower = real_prices * (1 + pred_min_pct)
        forecast_upper = real_prices * (1 + pred_max_pct)

        # 5. Vẽ Biểu Đồ
        plt.figure(figsize=(15, 7))

        # Vẽ giá thực tế
        plt.plot(dates, real_prices, label='Giá Thực Tế (Close)', color='black', linewidth=2)

        # Vẽ vùng mây dự báo (Confidence Interval)
        plt.fill_between(dates, forecast_lower, forecast_upper, color='green', alpha=0.2, label='Vùng Dự Báo (Min-Max)')

        # Vẽ biên trên và dưới
        plt.plot(dates, forecast_upper, color='green', linestyle='--', alpha=0.5, linewidth=1)
        plt.plot(dates, forecast_lower, color='red', linestyle='--', alpha=0.5, linewidth=1)

        plt.title(f'AI Vision: Vùng giá dự báo trong {days_to_plot} phiên gần nhất')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá Vàng ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == "__main__":
    viz = Visualizer()
    viz.plot_forecast()
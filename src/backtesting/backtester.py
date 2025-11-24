import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
from typing import Dict
from src.training.data_provider import DataProvider


class Backtester:
    def __init__(self, settings: Dict, initial_capital=10000):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.initial_capital = initial_capital

        model_name = settings['model']['name']
        self.model_path = os.path.join(settings['paths']['model_save'], f"{model_name}_best.keras")
        self.figures_dir = settings['paths']['figures_save']

        self.provider = DataProvider(settings)

    def run(self):
        self.logger.info("⏳ Đang tải dữ liệu kiểm thử (Test Set)...")
        try:
            _, _, X_test, y_test = self.provider.load_and_split()
        except Exception as e:
            self.logger.error(f"Lỗi load data: {e}")
            return

        self.logger.info("🧠 Đang load Model để backtest...")
        if not os.path.exists(self.model_path):
            self.logger.error(f"❌ Không tìm thấy model tại {self.model_path}")
            return

        model = tf.keras.models.load_model(self.model_path)
        # Dự đoán trên toàn bộ tập Test trước
        preds = model.predict([X_test['input_price'], X_test['input_macro']], verbose=0)

        pred_min = preds[0].flatten()
        pred_max = preds[1].flatten()

        # --- CHUẨN BỊ DỮ LIỆU THỰC TẾ ---
        df = pd.read_csv(self.provider.data_path, index_col=0, parse_dates=True)
        test_ratio = self.settings['processing']['test_size']
        window_size = self.settings['processing']['window_size']

        # Logic để khớp index với tập test (Lấy phần đuôi)
        real_data_slice = df.iloc[-len(pred_min):]

        # Lấy dữ liệu cần thiết
        all_test_dates = real_data_slice.index
        all_actual_returns = real_data_slice['Log_Return'].values
        all_price_history = real_data_slice['Gold_Close'].values

        # ==============================================================================
        # 👇 [MỚI] BỘ LỌC THỜI GIAN: CHỈ LẤY TỪ ĐẦU NĂM NAY 👇
        # ==============================================================================
        self.logger.info("📅 Đang lọc dữ liệu từ đầu năm đến nay...")

        # Cách 1: Tự động lấy năm hiện tại trên máy tính
        current_year = datetime.now().year
        start_date_filter = f"{current_year}-01-01"

        # Cách 2: Hoặc bạn có thể điền cứng ngày bạn muốn (Ví dụ data của bạn đang ở 2025)
        # start_date_filter = "2025-01-01"

        # Tạo mặt nạ lọc (Mask)
        mask = all_test_dates >= pd.Timestamp(start_date_filter)

        # Áp dụng bộ lọc
        test_dates = all_test_dates[mask]
        pred_min = pred_min[mask]
        pred_max = pred_max[mask]
        actual_returns = all_actual_returns[mask]
        price_history = all_price_history[mask]

        if len(test_dates) == 0:
            self.logger.warning(f"⚠️ Không có dữ liệu nào từ sau ngày {start_date_filter}. Kiểm tra lại file CSV!")
            return

        self.logger.info(f"✅ Đã lọc: Còn lại {len(test_dates)} phiên giao dịch để Backtest.")
        # ==============================================================================

        # --- LOGIC BACKTEST (Giữ nguyên) ---
        self.logger.info("💸 Đang mô phỏng giao dịch...")

        current_balance = self.initial_capital
        position = 0
        equity_curve = []

        for i in range(len(pred_min)):
            p_min = pred_min[i]
            p_max = pred_max[i]

            # Chiến thuật Mid-Point
            expected_return = (p_min + p_max) / 2

            if expected_return > 0.005:  # Kỳ vọng lãi > 0.2%
                signal = 1
            elif expected_return < -0.005:
                signal = -1
            else:
                signal = position

            if signal == 1:
                position = 1
            elif signal == -1:
                position = 0

            if position == 1:
                # Nếu không phải ngày cuối cùng thì mới tính lãi
                if i < len(actual_returns):
                    daily_return = actual_returns[i]
                    current_balance = current_balance * (1 + daily_return)

            equity_curve.append(current_balance)

        self.plot_results(test_dates, equity_curve, price_history)

    def plot_results(self, dates, strategy_equity, price_history):
        plt.figure(figsize=(14, 6))

        # Reset lại vốn Buy & Hold về mốc ban đầu tại thời điểm đầu năm nay
        # để so sánh công bằng
        initial_price = price_history[0]
        buy_hold_return = (price_history / initial_price) * self.initial_capital

        plt.plot(dates, buy_hold_return, label='Buy & Hold', color='gray', linestyle='--', alpha=0.6)
        plt.plot(dates, strategy_equity, label='AI Strategy', color='green', linewidth=2)

        # Format ngày tháng
        start_str = dates[0].strftime('%Y-%m-%d')
        end_str = dates[-1].strftime('%Y-%m-%d')

        plt.title(f'Backtest Hiệu Quả Đầu Tư ({start_str} -> {end_str})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylabel('Tài sản ($)')

        os.makedirs(self.figures_dir, exist_ok=True)
        save_path = os.path.join(self.figures_dir, "backtest_YTD.png")
        plt.savefig(save_path)
        self.logger.info(f"📉 Đã lưu biểu đồ Backtest tại: {save_path}")
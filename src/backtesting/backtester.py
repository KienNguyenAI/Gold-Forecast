import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import logging
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
        self.logger.info("⏳ Đang tải dữ liệu kiểm thử...")
        try:
            # for_training=False để lấy đầy đủ dữ liệu
            _, _, X_test, y_test = self.provider.load_and_split(for_training=False)
        except Exception as e:
            self.logger.error(f"Lỗi load data: {e}")
            return

        if not os.path.exists(self.model_path):
            self.logger.error(f"❌ Không tìm thấy model tại {self.model_path}")
            return

        self.logger.info("🧠 AI đang phân tích vùng giá...")
        model = tf.keras.models.load_model(self.model_path)
        preds = model.predict([X_test['input_price'], X_test['input_macro']], verbose=0)

        # Dự báo biên độ % (VD: -0.03 và +0.04)
        pred_min_pct = preds[0].flatten()
        pred_max_pct = preds[1].flatten()

        # --- CHUẨN BỊ DỮ LIỆU THỰC TẾ ---
        df = pd.read_csv(self.provider.data_path, index_col=0, parse_dates=True)

        # Lấy đoạn dữ liệu tương ứng với tập test
        # (Logic khớp index như cũ)
        real_data_slice = df.iloc[-len(pred_min_pct):]
        prices = real_data_slice['Gold_Close'].values
        dates = real_data_slice.index

        # --- CHIẾN THUẬT SNIPER: CẮT LỖ & CHỐT LỜI ---
        self.logger.info("💸 Đang chạy Backtest với chiến thuật Sniper (SL/TP)...")

        balance = self.initial_capital
        position = 0  # 0: Tiền mặt, 1: Đang giữ Vàng
        entry_price = 0

        # Lưu lịch sử để vẽ
        equity_curve = []
        trade_history = []  # Lưu điểm mua/bán để vẽ mũi tên

        for i in range(len(prices) - 1):
            current_price = prices[i]
            next_price = prices[i + 1]  # Giá ngày mai (để tính lãi lỗ thực tế)

            # 1. AI Dự báo vùng giá cho kỳ tới
            # Lưu ý: AI dự báo cho 30-60 ngày, nhưng ta dùng nó làm khung tham chiếu ngay lập tức
            ai_min_level = current_price * (1 + pred_min_pct[i])  # Điểm Cắt lỗ
            ai_max_level = current_price * (1 + pred_max_pct[i])  # Điểm Chốt lời

            trend = "UP" if (pred_min_pct[i] + pred_max_pct[i]) > 0 else "DOWN"

            # 2. LOGIC VÀO LỆNH (ENTRY)
            if position == 0:
                # Chỉ mua nếu Trend là Tăng
                if trend == "UP":
                    position = 1
                    entry_price = current_price
                    trade_history.append((dates[i], current_price, 'buy'))

            # 3. LOGIC THOÁT LỆNH (EXIT) - Dựa trên Min/Max của AI
            elif position == 1:
                # Kiểm tra giá ngày mai (giả lập diễn biến thị trường)

                # Kịch bản A: Chạm Đỉnh dự báo -> CHỐT LỜI
                if next_price >= ai_max_level:
                    position = 0
                    balance = balance * (next_price / entry_price)
                    trade_history.append((dates[i + 1], next_price, 'sell_tp'))  # TP: Take Profit

                # Kịch bản B: Thủng Đáy dự báo -> CẮT LỖ
                elif next_price <= ai_min_level:
                    position = 0
                    balance = balance * (next_price / entry_price)
                    trade_history.append((dates[i + 1], next_price, 'sell_sl'))  # SL: Stop Loss

                # Kịch bản C: Trend đảo chiều thành Giảm -> Thoát sớm
                elif trend == "DOWN":
                    position = 0
                    balance = balance * (next_price / entry_price)
                    trade_history.append((dates[i + 1], next_price, 'sell_trend'))

                # Nếu chưa chạm gì cả -> Giữ lệnh, cập nhật giá trị tài sản tạm tính
                else:
                    pass  # Hold

            # Cập nhật giá trị tài sản (Equity)
            if position == 1:
                current_equity = balance * (current_price / entry_price)
            else:
                current_equity = balance

            equity_curve.append(current_equity)

        # Thêm ngày cuối cùng
        equity_curve.append(balance)

        self.plot_sniper_results(dates, equity_curve, prices, trade_history)

    def plot_sniper_results(self, dates, strategy_equity, prices, trades):
        plt.figure(figsize=(14, 7))

        # 1. Vẽ đường cong vốn
        buy_hold = (prices / prices[0]) * self.initial_capital
        plt.plot(dates, buy_hold, label='Buy & Hold', color='gray', linestyle='--', alpha=0.5)
        plt.plot(dates, strategy_equity, label='AI Sniper Strategy', color='blue', linewidth=2)

        # 2. Vẽ điểm mua bán
        for date, price, type in trades:
            # Chuyển đổi giá sang tỉ lệ vốn để vẽ đúng vị trí trên trục Y
            # (Mẹo: Đây là vẽ tượng trưng, thực tế nên vẽ 2 subplot: Giá và Vốn riêng)
            pass

            # Tính chỉ số
        final_bal = strategy_equity[-1]
        profit = ((final_bal - self.initial_capital) / self.initial_capital) * 100

        # Drawdown
        equity_arr = np.array(strategy_equity)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - peak) / peak
        max_dd = np.min(drawdown) * 100

        plt.title(
            f'Chiến thuật Sniper (Dựa trên Min/Max Dự báo)\nLợi nhuận: {profit:.2f}% | Max Drawdown: {max_dd:.2f}%')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylabel('Tài sản ($)')

        os.makedirs(self.figures_dir, exist_ok=True)
        save_path = os.path.join(self.figures_dir, "sniper_backtest.png")
        plt.savefig(save_path)
        self.logger.info(f"📉 Đã lưu kết quả Sniper tại: {save_path}")

        print("\n" + "=" * 40)
        print(f"🔫 KẾT QUẢ CHIẾN THUẬT SNIPER")
        print(f"💰 Vốn cuối cùng: ${final_bal:,.2f}")
        print(f"📈 Lợi nhuận ròng: {profit:.2f}%")
        print(f"📉 Max Drawdown:  {max_dd:.2f}% (Rủi ro tối đa)")
        print(f"🔄 Tổng số lệnh:  {len(trades) // 2} vòng giao dịch")
        print("=" * 40 + "\n")
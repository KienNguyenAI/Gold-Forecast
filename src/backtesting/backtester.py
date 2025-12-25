import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import logging
import joblib
from typing import Dict
from src.training.data_provider import DataProvider


class Backtester:
    def __init__(self, settings: Dict, initial_capital=10000):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.initial_capital = initial_capital
        self.figures_dir = settings['paths']['figures_save']

        # --- CẤU HÌNH CHO MODEL 30 NGÀY ---
        if 'long_term' in settings['models_config']:
            self.model_conf = settings['models_config']['long_term']
        else:
            self.logger.warning("⚠️ Không thấy cấu hình 'long_term', dùng mặc định.")
            self.model_conf = settings['model']

        self.settings['processing']['window_size'] = self.model_conf['window_size']
        self.provider = DataProvider(self.settings)

    def run(self):
        self.logger.info(f"⏳ Backtest Model: {self.model_conf['name']} (Chiến thuật: AI Dynamic Floor)...")

        # 1. Load Data
        try:
            provider = DataProvider(self.settings)
            _, _, X_test, _ = provider.load_and_split(for_training=True)

            df_full = pd.read_csv(self.provider.data_path, index_col=0, parse_dates=True)
            test_len = len(X_test['input_price'])
            df_test = df_full.iloc[-test_len:]

            prices = df_test['Gold_Close'].values
            dates = df_test.index

        except Exception as e:
            self.logger.error(f"Lỗi load data: {e}")
            return

        # 2. Load Model
        model_path = os.path.join(self.settings['paths']['model_save'], f"{self.model_conf['name']}.keras")
        if not os.path.exists(model_path):
            self.logger.error(f"❌ Không tìm thấy model tại {model_path}")
            return

        model = tf.keras.models.load_model(model_path)
        preds = model.predict([X_test['input_price'], X_test['input_macro']], verbose=0)

        pred_min_pct = preds[0].flatten()
        pred_max_pct = preds[1].flatten()

        # --- CHIẾN THUẬT: AI DYNAMIC FLOOR (SÀN ĐỠ GIÁ) ---
        self.logger.info("💸 Đang chạy chiến thuật AI Dynamic Floor (Bám sát Trend)...")

        balance = self.initial_capital
        position = 0  # 0: Cash, 1: Gold
        entry_price = 0
        shares = 0

        # Biến Stoploss động
        dynamic_sl = 0

        equity_curve = []
        trade_history = []

        for i in range(len(prices) - 1):
            current_price = prices[i]

            # Tính Equity
            if position == 1:
                current_equity = shares * current_price
            else:
                current_equity = balance
            equity_curve.append(current_equity)

            # --- LOGIC ---

            # Tính mức "Sàn" mà AI dự báo cho 30 ngày tới
            # Sàn AI = Giá hiện tại + % Min mà AI dự báo
            # Ví dụ: Giá 2000, AI báo Min -2% -> Sàn = 1960
            ai_floor_price = current_price * (1 + pred_min_pct[i])

            # 1. TÌM CƠ HỘI MUA (ENTRY)
            if position == 0:
                # Điều kiện mua:
                # 1. Tiềm năng tăng giá (Max) > 1.5%
                # 2. Rủi ro (Min) không quá sâu (> -4%)
                # 3. Quan trọng: AI phải dự báo xu hướng dương (Min + Max > 0)
                if pred_max_pct[i] > 0.015 and pred_min_pct[i] > -0.04:
                    position = 1
                    entry_price = current_price
                    shares = balance / current_price
                    balance = 0

                    # Đặt SL ban đầu ngay tại Sàn AI (trừ hao thêm 0.5% để tránh quét)
                    dynamic_sl = ai_floor_price * 0.995

                    trade_history.append((dates[i], current_price, 'buy'))

            # 2. QUẢN LÝ LỆNH (HOLDING)
            elif position == 1:

                # --- CẬP NHẬT STOPLOSS (TRAILING UP) ---
                # Nếu Sàn AI hôm nay cao hơn mức SL cũ -> Dời SL lên
                # Nguyên tắc: Chỉ dời lên, không bao giờ dời xuống
                if ai_floor_price > dynamic_sl:
                    # Tuy nhiên, không dời quá sát giá hiện tại (giữ khoảng cách tối thiểu 1.5%)
                    # Để giá có không gian thở
                    max_sl_allowed = current_price * 0.985
                    new_sl = min(ai_floor_price, max_sl_allowed)

                    if new_sl > dynamic_sl:
                        dynamic_sl = new_sl

                # --- ĐIỀU KIỆN BÁN ---
                is_stop_loss = current_price <= dynamic_sl

                # Điều kiện bán khẩn cấp: Nếu AI bỗng dưng dự báo sập mạnh (Max < -1%)
                ai_collapse_signal = pred_max_pct[i] < -0.01

                if is_stop_loss:
                    balance = shares * current_price
                    shares = 0
                    position = 0

                    reason = 'sell_profit' if current_price > entry_price else 'sell_loss'
                    trade_history.append((dates[i], current_price, reason))

                elif ai_collapse_signal:
                    balance = shares * current_price
                    shares = 0
                    position = 0
                    trade_history.append((dates[i], current_price, 'sell_ai_panic'))

        # --- XỬ LÝ CUỐI KỲ ---
        if position == 1:
            final_equity = shares * prices[-1]
            trade_history.append((dates[-1], prices[-1], 'force_close'))
        else:
            final_equity = balance

        equity_curve.append(final_equity)

        self.plot_sniper_results(dates, equity_curve, prices, trade_history, self.model_conf['name'])

    def plot_sniper_results(self, dates, strategy_equity, prices, trades, model_name):
        plt.figure(figsize=(14, 7))

        # 1. Buy & Hold
        buy_hold_return_pct = ((prices[-1] - prices[0]) / prices[0]) * 100
        buy_hold_final = self.initial_capital * (prices[-1] / prices[0])
        buy_hold_curve = (prices / prices[0]) * self.initial_capital

        plt.plot(dates, buy_hold_curve, label=f'Buy & Hold ({buy_hold_return_pct:.2f}%)',
                 color='gray', linestyle='--', alpha=0.6)

        # 2. AI Strategy
        final_bal = strategy_equity[-1]
        strategy_profit_pct = ((final_bal - self.initial_capital) / self.initial_capital) * 100

        plt.plot(dates, strategy_equity, label=f'AI Dynamic Floor ({strategy_profit_pct:.2f}%)',
                 color='#009688', linewidth=2)

        # 3. Điểm mua bán
        for date, price, action in trades:
            if action == 'buy':
                plt.scatter(date, strategy_equity[np.where(dates == date)[0][0]],
                            marker='^', color='green', s=80, zorder=5)
            elif 'sell' in action or 'force' in action:
                color = 'gold' if 'profit' in action else 'red'
                plt.scatter(date, strategy_equity[np.where(dates == date)[0][0]],
                            marker='v', color=color, s=80, zorder=5)

        # 4. Drawdown
        equity_arr = np.array(strategy_equity)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - peak) / peak
        max_dd = np.min(drawdown) * 100

        plt.title(
            f'Backtest Model: {model_name} (Dynamic Floor)\nProfit: {strategy_profit_pct:.2f}% | Max DD: {max_dd:.2f}%')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        os.makedirs(self.figures_dir, exist_ok=True)
        save_path = os.path.join(self.figures_dir, "backtest_30days.png")
        plt.savefig(save_path)
        self.logger.info(f"📉 Đã lưu biểu đồ Backtest tại: {save_path}")

        # --- BÁO CÁO ---
        num_trades = len([t for t in trades if t[2] == 'buy'])

        print("\n" + "=" * 60)
        print(f"📊 KẾT QUẢ BACKTEST (AI DYNAMIC FLOOR)")
        print(f"AI TRADING (Model: {model_name}):")
        print(f"   - Vốn cuối: ${final_bal:,.2f} ({strategy_profit_pct:+.2f}%)")
        print(f"   - Max Drawdown: {max_dd:.2f}%")
        print(f"   - Tổng số lệnh: {num_trades}")

        alpha = strategy_profit_pct - buy_hold_return_pct
        print("=" * 60 + "\n")
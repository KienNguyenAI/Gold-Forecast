import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from training.data_provider import DataProvider


class Backtester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.model_path = "models/best_gold_model.keras"
        self.provider = DataProvider(window_size=60)

    def run(self):
        print("⏳ Đang tải dữ liệu kiểm thử (Test Set)...")
        # Lấy dữ liệu Test (Dữ liệu AI chưa từng nhìn thấy)
        _, _, X_test, y_test = self.provider.load_and_split(train_ratio=0.8)

        print("🧠 Đang chạy Model để dự đoán quá khứ...")
        model = tf.keras.models.load_model(self.model_path)

        # Dự đoán hàng loạt
        preds = model.predict(X_test, verbose=1)

        # Tách 2 đầu ra
        # preds là list [pred_min, pred_max]
        pred_min = preds[0].flatten()  # Output 1: Min Change
        pred_max = preds[1].flatten()  # Output 2: Max Change

        # Target thực tế (để tính lợi nhuận nếu mua)
        # Lưu ý: y_test là {output_min, output_max}, nhưng ta cần giá thực tế để tính lãi
        # Để đơn giản, ta giả định lợi nhuận của Buy & Hold là trung bình của biến động thực tế
        # Trong thực tế, ta cần cột 'Close' gốc, nhưng ở đây ta dùng mẹo xấp xỉ:
        # Lợi nhuận thực tế xấp xỉ = (Actual_Max + Actual_Min) / 2 (Trung bình biến động tháng đó)
        # Hoặc chính xác hơn: Ta cần lấy lại Log Return thực tế.

        # Tuy nhiên, để chính xác nhất, ta sẽ so sánh chiến lược với Buy & Hold
        # Ta sẽ giả lập PnL dựa trên Signals

        capital = [self.initial_capital]
        signals = []  # 1: Buy, -1: Sell, 0: Hold

        print("💸 Đang mô phỏng giao dịch...")

        # Giả sử mỗi lần trade giữ lệnh 1 tháng (22 ngày) hoặc đến khi đảo chiều
        # Ở đây làm simplified backtest: Cộng dồn lợi nhuận nếu dự đoán đúng hướng

        # Lấy dữ liệu Close gốc tương ứng với tập Test để tính PnL thật
        df = pd.read_csv(self.provider.data_path, index_col=0, parse_dates=True)
        split_idx = int(len(df) * 0.8)
        test_dates = df.index[split_idx + 60:]  # +60 vì window size

        # Lấy % thay đổi giá thực tế của ngày hôm sau (để tính lãi lỗ từng ngày)
        # Shift(-1) để biết mua hôm nay, mai lãi bao nhiêu
        actual_returns = df['Log_Return'].iloc[split_idx + 60:].values

        # Đảm bảo độ dài khớp nhau
        limit = min(len(pred_min), len(actual_returns))

        current_balance = self.initial_capital
        position = 0  # 0: Cash, 1: Long

        equity_curve = []

        for i in range(limit):
            # 1. Logic Ra Quyết Định (SỬA LẠI ĐOẠN NÀY)
            p_min = pred_min[i]
            p_max = pred_max[i]

            # --- CHIẾN THUẬT MỚI: MID-POINT STRATEGY ---
            # Tính trung bình cộng của Đáy và Đỉnh dự báo
            # Ví dụ: Min là -1%, Max là +5% -> Trung bình là +2% -> MUA
            expected_return = (p_min + p_max) / 2

            # Ngưỡng kích hoạt mua: Chỉ cần kỳ vọng lãi > 0.2% (để bù phí)
            if expected_return > 0.002:
                signal = 1  # Buy

            # Ngưỡng bán: Nếu kỳ vọng lỗ hoặc Max quá thấp
            elif expected_return < -0.002:
                signal = -1  # Sell / Cash out

            else:
                # Vùng trung tính: Giữ nguyên trạng thái đang có (Trend Following)
                # Nếu đang cầm hàng thì giữ, đang cầm tiền thì thôi
                signal = position

                # 2. Thực hiện lệnh (Giữ nguyên)
            if signal == 1:
                position = 1
            elif signal == -1:
                position = 0

            # 3. Tính lãi/lỗ (Giữ nguyên)
            if position == 1:
                daily_return = actual_returns[i]
                current_balance = current_balance * (1 + daily_return)

            equity_curve.append(current_balance)

        # Vẽ biểu đồ so sánh
        self.plot_results(test_dates[:limit], equity_curve, df['Gold_Close'].iloc[split_idx + 60:].values[:limit])

    def plot_results(self, dates, strategy_equity, price_history):
        plt.figure(figsize=(14, 6))

        # Chuẩn hóa giá vàng về cùng mốc vốn $10,000 để so sánh
        buy_hold_return = (price_history / price_history[0]) * self.initial_capital

        plt.plot(dates, buy_hold_return, label='Buy & Hold (Mua xong để đấy)', color='gray', linestyle='--', alpha=0.6)
        plt.plot(dates, strategy_equity, label='AI Strategy (Safe Entry)', color='green', linewidth=2)

        plt.title(f'Backtest: AI Strategy vs Buy & Hold (Vốn khởi đầu ${self.initial_capital})')
        plt.xlabel('Thời gian')
        plt.ylabel('Tài sản ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Tính lợi nhuận cuối cùng
        final_balance = strategy_equity[-1]
        profit_pct = ((final_balance - self.initial_capital) / self.initial_capital) * 100

        print("\n" + "=" * 40)
        print(f"🏁 KẾT QUẢ BACKTEST:")
        print(f"💰 Vốn ban đầu: ${self.initial_capital}")
        print(f"💰 Vốn kết thúc: ${final_balance:.2f}")
        print(f"📈 Lợi nhuận ròng: {profit_pct:.2f}%")
        print("=" * 40)

        if final_balance > buy_hold_return[-1]:
            print("🌟 TUYỆT VỜI! AI đã đánh bại thị trường (Beat the Market).")
        else:
            print("🐢 AI an toàn nhưng lợi nhuận thấp hơn Buy&Hold (Điều bình thường với chiến lược quản trị rủi ro).")

        plt.show()


if __name__ == "__main__":
    bot = Backtester()
    bot.run()
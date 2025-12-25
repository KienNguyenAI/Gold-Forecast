import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict
from src.training.data_provider import DataProvider


class ModelEvaluator:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.models_cfg = settings.get('models_config')  # Lấy config đa model
        self.figures_dir = settings['paths']['figures_save']

        # Đường dẫn data gốc để tham chiếu giá thật
        self.raw_data_path = os.path.join(settings['paths']['processed_data'], "gold_processed_features.csv")

    def run(self):
        self.logger.info("📊 BẮT ĐẦU ĐÁNH GIÁ HIỆU SUẤT ĐA MÔ HÌNH (EVALUATION)...")

        if not self.models_cfg:
            self.logger.error("❌ Không tìm thấy 'models_config' trong settings.")
            return

        # Vòng lặp đánh giá từng model
        for key, config in self.models_cfg.items():
            self._evaluate_single_model(key, config)

    def _evaluate_single_model(self, model_key, model_conf):
        print("\n" + "#" * 60)
        print(f"🧩 ĐANG ĐÁNH GIÁ MODEL: {model_key.upper()} ({model_conf['name']})")
        print("#" * 60)

        # 1. Cập nhật Settings tạm thời để DataProvider lấy đúng Window Size
        # (Short dùng 30, Long dùng 90...)
        self.settings['processing']['window_size'] = model_conf['window_size']
        self.settings['model'] = model_conf  # Để tương thích nếu DataProvider cần

        # 2. Load Data Test riêng cho model này
        try:
            provider = DataProvider(self.settings)
            # for_training=True để lấy tập X_test, y_test đã split
            _, _, X_test, y_test = provider.load_and_split(for_training=True)

            # Load dữ liệu gốc để lấy giá Close (USD) tương ứng với tập Test
            if not os.path.exists(self.raw_data_path):
                self.logger.error(f"Thiếu file data: {self.raw_data_path}")
                return

            df_full = pd.read_csv(self.raw_data_path, index_col=0, parse_dates=True)

            # Cắt lấy phần giá gốc tương ứng với y_test
            # (Lưu ý: y_test là phần cuối của dataset)
            test_len = len(y_test['output_min'])
            df_test_raw = df_full.iloc[-test_len:]
            current_prices = df_test_raw['Gold_Close'].values

        except Exception as e:
            self.logger.error(f"Lỗi load data cho {model_key}: {e}")
            return

        # 3. Load Model
        model_path = os.path.join(self.settings['paths']['model_save'], f"{model_conf['name']}.keras")
        if not os.path.exists(model_path):
            self.logger.error(f"❌ Không tìm thấy model tại: {model_path}")
            return

        try:
            # compile=False để tránh lỗi custom_loss nếu không cần train tiếp
            model = tf.keras.models.load_model(model_path, compile=False)
            preds = model.predict([X_test['input_price'], X_test['input_macro']], verbose=0)
        except Exception as e:
            self.logger.error(f"Lỗi khi dự báo model {model_key}: {e}")
            return

        # 4. Chuẩn bị dữ liệu so sánh
        pred_min_pct = preds[0].flatten()
        pred_max_pct = preds[1].flatten()
        actual_min_pct = y_test['output_min']
        actual_max_pct = y_test['output_max']

        # Quy đổi % sang USD
        pred_price_min = current_prices * (1 + pred_min_pct)
        pred_price_max = current_prices * (1 + pred_max_pct)
        actual_price_min = current_prices * (1 + actual_min_pct)
        actual_price_max = current_prices * (1 + actual_max_pct)

        # Tính xu hướng trung bình thực tế
        actual_avg_pct = (actual_min_pct + actual_max_pct) / 2

        # 5. Tính toán & In báo cáo
        self._calculate_metrics(
            model_key,
            actual_price_min, actual_price_max,
            pred_price_min, pred_price_max,
            actual_avg_pct,
            pred_min_pct, pred_max_pct
        )

    def _calculate_metrics(self, model_name, act_min, act_max, pred_min, pred_max, act_avg_pct, pred_min_pct,
                           pred_max_pct):

        # --- A. REGRESSION METRICS (USD) ---
        mae_min = mean_absolute_error(act_min, pred_min)
        mae_max = mean_absolute_error(act_max, pred_max)
        avg_mae_usd = (mae_min + mae_max) / 2

        rmse = np.sqrt(mean_squared_error(
            np.concatenate([act_min, act_max]),
            np.concatenate([pred_min, pred_max])
        ))

        # MAPE
        mape_min = np.mean(np.abs((act_min - pred_min) / act_min)) * 100
        mape_max = np.mean(np.abs((act_max - pred_max) / act_max)) * 100
        avg_mape = (mape_min + mape_max) / 2

        # --- B. DIRECTION ACCURACY (Xu hướng) ---
        pred_avg_pct = (pred_min_pct + pred_max_pct) / 2
        act_trend = np.sign(act_avg_pct + 1e-9)
        pred_trend = np.sign(pred_avg_pct + 1e-9)
        accuracy = np.mean(act_trend == pred_trend) * 100

        # --- C. RANGE EFFICIENCY ---
        act_spread = np.mean(act_max - act_min)
        pred_spread = np.mean(pred_max - pred_min)
        spread_ratio = pred_spread / (act_spread + 1e-9)

        # --- REPORT ---
        print(f"\n🔍 KẾT QUẢ CHI TIẾT: {model_name.upper()}")
        print("-" * 40)

        print("1. ĐỘ CHÍNH XÁC VỀ GIÁ (Price Accuracy):")
        print(f"   - MAE (Sai lệch trung bình):  ${avg_mae_usd:.2f}")
        print(f"   - RMSE:                       ${rmse:.2f}")
        print(f"   - MAPE:                       {avg_mape:.2f}%")

        print("\n2. KHẢ NĂNG BẮT XU HƯỚNG (Trend Prediction):")
        print(f"   - Accuracy:                   {accuracy:.2f}%")

        print("\n3. HIỆU SUẤT ĐỘ RỘNG (Range Efficiency):")
        print(f"   - Độ rộng TB Thực tế:         ${act_spread:.2f}")
        print(f"   - Độ rộng TB Dự báo:          ${pred_spread:.2f}")
        print(f"   - Ratio (Dự báo/Thực tế):     {spread_ratio:.2f}x")

        if spread_ratio > 1.2:
            print("   -> Nhận xét: Vùng dự báo hơi RỘNG (Thận trọng).")
        elif spread_ratio < 0.8:
            print("   -> Nhận xét: Vùng dự báo hơi HẸP (Tự tin/Rủi ro).")
        else:
            print("   -> Nhận xét: Vùng dự báo CÂN BẰNG.")

        print("-" * 40 + "\n")

        # Vẽ Scatter Plot riêng cho model này
        self._plot_scatter(act_max, pred_max, f"Scatter - {model_name} (Max Price)")

    def _plot_scatter(self, y_true, y_pred, title):
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5, color='blue')

        # Đường chuẩn 45 độ
        lims = [
            np.min([plt.xlim(), plt.ylim()]),
            np.max([plt.xlim(), plt.ylim()]),
        ]
        plt.plot(lims, lims, 'r-', alpha=0.75, zorder=0, label="Perfect Prediction")

        plt.title(title)
        plt.xlabel('Giá Thực tế (USD)')
        plt.ylabel('Giá Dự báo (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        os.makedirs(self.figures_dir, exist_ok=True)
        # Lưu file với tên model
        safe_title = title.replace(" ", "_").replace("(", "").replace(")", "").lower()
        save_path = os.path.join(self.figures_dir, f"{safe_title}.png")
        plt.savefig(save_path)
        plt.close()
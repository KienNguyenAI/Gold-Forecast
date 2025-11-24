import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from training.data_provider import DataProvider
from datetime import timedelta


def predict_future():
    print("🔮 ĐANG KHỞI ĐỘNG HỆ THỐNG DỰ BÁO...")

    # 1. Load các công cụ
    # Load Model
    model_path = "models/best_gold_model.keras"
    if not os.path.exists(model_path):
        print("❌ Chưa có model! Hãy chạy main_train.py trước.")
        return
    model = tf.keras.models.load_model(model_path)

    # Load Scalers (Để giải mã dữ liệu)
    try:
        scaler_tech = joblib.load("models/scaler_tech.pkl")
        scaler_macro = joblib.load("models/scaler_macro.pkl")
    except:
        print("❌ Thiếu file Scaler. Hãy chạy main_train.py trước.")
        return

    # 2. Lấy dữ liệu mới nhất
    # Ta dùng lại class DataProvider nhưng chỉ để lấy raw data
    provider = DataProvider(window_size=60)
    # Load toàn bộ dữ liệu (đã qua xử lý sơ bộ ở main_process)
    df = pd.read_csv(provider.data_path, index_col=0, parse_dates=True)

    print(f"📅 Dữ liệu cập nhật đến ngày: {df.index[-1].date()}")

    # 3. Chuẩn bị Input cho ngày hôm nay (Lấy 60 dòng cuối cùng)
    last_60_days = df.iloc[-60:]

    # Scale dữ liệu (Giống hệt lúc train)
    input_tech_raw = last_60_days[provider.tech_cols]
    input_macro_raw = last_60_days[
        provider.macro_cols]  # Lấy dòng cuối (hoặc cả chuỗi cũng dc, nhưng code train lấy dòng cuối)

    input_tech_scaled = scaler_tech.transform(input_tech_raw)
    input_macro_scaled = scaler_macro.transform(input_macro_raw)

    # Reshape sang 3D cho LSTM: (1 mẫu, 60 ngày, 5 feature)
    X_tech = np.array([input_tech_scaled])

    # Input Macro: Lấy dòng cuối cùng (ngày mới nhất)
    # Shape: (1 mẫu, 4 feature)
    X_macro = np.array([input_macro_scaled[-1]])

    # 4. DỰ ĐOÁN
    print("⏳ AI đang suy nghĩ...")
    pred_min_change, pred_max_change = model.predict(
        {'input_price': X_tech, 'input_macro': X_macro},
        verbose=0
    )

    # 5. Giải mã kết quả (Từ % -> Giá USD)
    current_price = df['Gold_Close'].iloc[-1]

    # Model trả về mảng 2 chiều [[value]], ta lấy value ra
    pct_min = pred_min_change[0][0]
    pct_max = pred_max_change[0][0]

    predicted_min = current_price * (1 + pct_min)
    predicted_max = current_price * (1 + pct_max)

    print("\n" + "=" * 40)
    print(f"💰 GIÁ VÀNG HIỆN TẠI: ${current_price:.2f}")
    print("=" * 40)
    print(f"🎯 DỰ BÁO VÙNG GIÁ TRONG 30 NGÀY TỚI:")
    print(f"   📉 Đáy thấp nhất (Min): ${predicted_min:.2f} ({pct_min * 100:+.2f}%)")
    print(f"   📈 Đỉnh cao nhất (Max): ${predicted_max:.2f} ({pct_max * 100:+.2f}%)")
    print("=" * 40)

    # Insight khuyến nghị
    spread = predicted_max - predicted_min
    print(f"⚠️ Biên độ dao động dự kiến: ${spread:.2f}")

    if pct_min > 0:
        print("🚀 TÍN HIỆU: UPTREND MẠNH (Cả đáy dự báo cũng cao hơn giá hiện tại). -> MUA")
    elif pct_max < 0:
        print("🔻 TÍN HIỆU: DOWNTREND (Cả đỉnh dự báo cũng thấp hơn giá hiện tại). -> BÁN/SHORT")
    else:
        print("↔️ TÍN HIỆU: SIDEWAY/BIẾN ĐỘNG (Giá sẽ chạy trong vùng trên). -> Mua thấp Bán cao.")


if __name__ == "__main__":
    predict_future()
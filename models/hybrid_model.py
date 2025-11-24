import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.models import Model

class GoldPriceModel:
    def __init__(self, window_size=60, n_features_price=6, n_features_macro=4):
        """
        Khởi tạo thông số mô hình.
        :param window_size: Độ dài chuỗi giá quá khứ (VD: 60 ngày)
        :param n_features_price: Số lượng chỉ số kỹ thuật (Close, RSI, Volatility...)
        :param n_features_macro: Số lượng chỉ số vĩ mô (CPI, US10Y...)
        """
        self.window_size = window_size
        self.n_features_price = n_features_price
        self.n_features_macro = n_features_macro
    def build_model(self):
        # --- NHÁNH 1: Xử lý Chuỗi Thời Gian (Price History) ---
        input_price = Input(shape=(self.window_size, self.n_features_price), name='input_price')
        x1 = LSTM(64, return_sequences=True, activation='tanh')(input_price)
        x1 = Dropout(0.2)(x1)
        x1 = LSTM(32, return_sequences=False, activation='tanh')(x1)
        x1 = BatchNormalization()(x1)

        # --- NHÁNH 2: Xử lý Vĩ Mô (Macro Data) ---
        input_macro = Input(shape=(self.n_features_macro,), name='input_macro')
        x2 = Dense(32, activation='relu')(input_macro)
        x2 = Dropout(0.2)(x2)
        x2 = Dense(16, activation='relu')(x2)

        # --- GHÉP 2 NHÁNH (CONCATENATE) ---
        combined = Concatenate()([x1, x2])
        z = Dense(64, activation='relu')(combined)
        z = Dropout(0.2)(z)

        # --- OUTPUT LAYERS (Đầu ra) ---
        # Đầu ra 1: Dự đoán % thay đổi giá Thấp Nhất (Min)
        # Linear activation vì giá trị có thể là số âm hoặc dương bất kỳ
        out_min = Dense(1, activation='linear', name='output_min')(z)
        # Đầu ra 2: Dự đoán % thay đổi giá Cao Nhất (Max)
        out_max = Dense(1, activation='linear', name='output_max')(z)

        # Tạo Model tổng thể
        model = Model(inputs=[input_price, input_macro], outputs=[out_min, out_max])

        return model





if __name__ == "__main__":
    builder = GoldPriceModel()
    model = builder.build_model()
    model.summary()
    print("✅ Build model thành công!")
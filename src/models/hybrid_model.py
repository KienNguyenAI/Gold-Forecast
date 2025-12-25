from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from typing import Dict, Tuple
import logging


class GoldPriceModel:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.model_conf = settings['model']
        self.process_conf = settings['processing']

        self.window_size = self.process_conf['window_size']
        self.dropout_rate = self.model_conf.get('dropout', 0.2)
        self.hidden_dim = self.model_conf.get('hidden_dim', 64)

        self.n_features_price = self.model_conf.get('input_dim', 6)

    def build_model(self, input_shape_price: Tuple[int, int], input_shape_macro: Tuple[int,]):
        self.logger.info(f"Đang xây dựng model Hybrid: LSTM ({input_shape_price}) + Dense ({input_shape_macro})")

        # --- NHÁNH 1: Xử lý Chuỗi Thời Gian (Price History) ---
        input_price = Input(shape=input_shape_price, name='input_price')

        x1 = LSTM(self.hidden_dim, return_sequences=True, activation='tanh')(input_price)
        x1 = Dropout(self.dropout_rate)(x1)
        x1 = LSTM(int(self.hidden_dim / 2), return_sequences=False, activation='tanh')(x1)
        x1 = BatchNormalization()(x1)

        # --- NHÁNH 2: Xử lý Vĩ Mô (Macro Data) ---
        input_macro = Input(shape=input_shape_macro, name='input_macro')

        x2 = Dense(int(self.hidden_dim / 2), activation='relu')(input_macro)
        x2 = Dropout(self.dropout_rate)(x2)
        x2 = Dense(16, activation='relu')(x2)

        # --- GHÉP 2 NHÁNH (CONCATENATE) ---
        combined = Concatenate()([x1, x2])
        z = Dense(self.hidden_dim, activation='relu')(combined)
        z = Dropout(self.dropout_rate)(z)

        # --- OUTPUT LAYERS (Đầu ra) ---
        # Output 1: Min Change
        out_min = Dense(1, activation='linear', name='output_min')(z)
        # Output 2: Max Change
        out_max = Dense(1, activation='linear', name='output_max')(z)
        model = Model(inputs=[input_price, input_macro], outputs=[out_min, out_max])

        self.logger.info("Build model kiến trúc thành công.")
        return model


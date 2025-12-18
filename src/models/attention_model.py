from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, BatchNormalization, Layer, GlobalAveragePooling1D, Reshape, Multiply, Permute, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from typing import Dict, Tuple
import logging
import tensorflow as tf


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='normal')
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros')
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x shape: (batch_size, time_steps, features)
        # e = tanh(dot(x, W) + b)
        e = K.tanh(K.dot(x, self.W) + self.b)
        # a = softmax(e)
        a = K.softmax(e, axis=1)
        # output = x * a
        output = x * a
        # sum over time steps
        return K.sum(output, axis=1)


class GoldPriceAttentionModel:
    def __init__(self, settings: Dict):
        """
        :param settings: Dictionary chứa toàn bộ config (settings.yaml)
        """
        self.logger = logging.getLogger(__name__)
        self.model_conf = settings['model']
        self.process_conf = settings['processing']

        self.window_size = self.process_conf['window_size']
        self.dropout_rate = self.model_conf.get('dropout', 0.2)
        self.hidden_dim = self.model_conf.get('hidden_dim', 64)

        self.n_features_price = self.model_conf.get('input_dim', 6)

    def build_model(self, input_shape_price: Tuple[int, int], input_shape_macro: Tuple[int,]):
        """
        :param input_shape_price: Shape của nhánh giá (window_size, n_features)
        :param input_shape_macro: Shape của nhánh vĩ mô (n_features,)
        """
        self.logger.info(f"Đang xây dựng model Attention: LSTM+Attention ({input_shape_price}) + Dense ({input_shape_macro})")

        # --- NHÁNH 1: Xử lý Chuỗi Thời Gian (Price History) với Attention ---
        input_price = Input(shape=input_shape_price, name='input_price')

        # LSTM trả về sequences để Attention layer có thể đánh trọng số từng bước thời gian
        x1 = LSTM(self.hidden_dim, return_sequences=True, activation='tanh')(input_price)
        x1 = Dropout(self.dropout_rate)(x1)
        
        # Attention Layer
        # Cách 1: Custom Layer (đơn giản)
        # context_vector = AttentionLayer()(x1)
        
        # Cách 2: Dùng cơ chế Attention cơ bản với Keras layers
        # Tính attention scores
        attention = Dense(1, activation='tanh')(x1) # (batch, steps, 1)
        attention = Reshape((input_shape_price[0],))(attention) # (batch, steps)
        attention = tf.keras.layers.Activation('softmax')(attention) # (batch, steps) - Trọng số cho mỗi bước
        
        # Áp dụng trọng số vào LSTM output
        # Repeat vector để nhân với từng feature
        attention_expanded = Reshape((input_shape_price[0], 1))(attention) # (batch, steps, 1)
        weighted_x1 = Multiply()([x1, attention_expanded]) # (batch, steps, features)
        
        # Tổng hợp lại thành 1 vector ngữ cảnh (Context Vector)
        context_vector = GlobalAveragePooling1D()(weighted_x1) # (batch, features)
        # Hoặc dùng Sum: context_vector = Lambda(lambda x: K.sum(x, axis=1))(weighted_x1)

        x1_out = BatchNormalization()(context_vector)

        # --- NHÁNH 2: Xử lý Vĩ Mô (Macro Data) ---
        input_macro = Input(shape=input_shape_macro, name='input_macro')

        x2 = Dense(int(self.hidden_dim / 2), activation='relu')(input_macro)
        x2 = Dropout(self.dropout_rate)(x2)
        x2 = Dense(16, activation='relu')(x2)

        # --- GHÉP 2 NHÁNH (CONCATENATE) ---
        combined = Concatenate()([x1_out, x2])
        z = Dense(self.hidden_dim, activation='relu')(combined)
        z = Dropout(self.dropout_rate)(z)

        # --- OUTPUT LAYERS (Đầu ra) ---
        # Output 1: Min Change
        out_min = Dense(1, activation='linear', name='output_min')(z)
        # Output 2: Max Change
        out_max = Dense(1, activation='linear', name='output_max')(z)

        # Tạo Model tổng thể
        model = Model(inputs=[input_price, input_macro], outputs=[out_min, out_max])

        self.logger.info("Build model Attention kiến trúc thành công.")
        return model

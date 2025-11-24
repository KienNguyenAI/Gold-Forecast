import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import logging
import matplotlib.pyplot as plt
from typing import Dict

# Import class từ trong src
from .data_provider import DataProvider
# 👇 Lưu ý: Import từ module models (Hybrid Model)
from src.models.hybrid_model import GoldPriceModel


class ModelTrainer:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.settings = settings

        # Lấy tham số training từ config
        self.train_conf = settings['training']
        self.epochs = self.train_conf['epochs']
        self.batch_size = self.train_conf['batch_size']

        # Đường dẫn lưu model
        save_dir = settings['paths']['model_save']
        os.makedirs(save_dir, exist_ok=True)
        # Lưu tên model có version để dễ quản lý
        model_name = settings['model'].get('name', 'model')
        self.model_save_path = os.path.join(save_dir, f"{model_name}_best.keras")
        self.figures_dir = settings['paths']['figures_save']

    def train(self):
        self.logger.info("🚀 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN...")

        # 1. Chuẩn bị dữ liệu
        provider = DataProvider(self.settings)
        X_train, y_train, X_test, y_test = provider.load_and_split()

        # Lưu scaler
        provider.save_scalers()

        # 2. Xây dựng mô hình
        # Lấy shape động
        n_features_price = X_train['input_price'].shape[2]
        n_features_macro = X_train['input_macro'].shape[1]

        self.logger.info(f"📊 Input Features: Price={n_features_price}, Macro={n_features_macro}")

        # Khởi tạo Builder (truyền setting vào)
        builder = GoldPriceModel(self.settings)

        # Build với shape cụ thể
        model = builder.build_model(
            input_shape_price=(X_train['input_price'].shape[1], n_features_price),
            input_shape_macro=(n_features_macro,)
        )

        # 3. Compile
        learning_rate = self.train_conf.get('learning_rate', 0.001)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={'output_min': 'mse', 'output_max': 'mse'},
            metrics={'output_min': 'mae', 'output_max': 'mae'}
        )

        # 4. Callbacks
        callbacks = [
            ModelCheckpoint(
                self.model_save_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=self.train_conf.get('patience', 10),
                restore_best_weights=True,
                verbose=1
            )
        ]

        # 5. Training Loop
        history = model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_test, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.logger.info("✅ HUẤN LUYỆN HOÀN TẤT!")
        self.plot_history(history)

        return self.model_save_path

    def plot_history(self, history):
        """Vẽ biểu đồ và lưu vào file ảnh thay vì chỉ show"""
        os.makedirs(self.figures_dir, exist_ok=True)

        plt.figure(figsize=(12, 5))

        # Plot 1: Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Total Loss')
        plt.legend()

        # Plot 2: MAE
        plt.subplot(1, 2, 2)
        # Kiểm tra key trong history (vì tf version khác nhau có thể đổi tên)
        if 'val_output_min_mae' in history.history:
            plt.plot(history.history['val_output_min_mae'], label='Min MAE')
            plt.plot(history.history['val_output_max_mae'], label='Max MAE')
        else:
            # Fallback cho tên key khác
            pass

        plt.title('Validation MAE')
        plt.legend()
        plt.tight_layout()

        # Lưu ảnh
        save_path = os.path.join(self.figures_dir, "training_history.png")
        plt.savefig(save_path)
        self.logger.info(f"📉 Đã lưu biểu đồ training tại: {save_path}")
        plt.close()  # Đóng plot để giải phóng mem
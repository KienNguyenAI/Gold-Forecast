import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import logging
import matplotlib.pyplot as plt
from typing import Dict

import mlflow
import mlflow.tensorflow

from .data_provider import DataProvider
from src.models.hybrid_model import GoldPriceModel


class ModelTrainer:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.settings = settings

        self.train_conf = settings['training']
        self.epochs = self.train_conf['epochs']
        self.batch_size = self.train_conf['batch_size']

        save_dir = settings['paths']['model_save']
        os.makedirs(save_dir, exist_ok=True)
        model_name = settings['model'].get('name', 'model')
        self.model_save_path = os.path.join(save_dir, f"{model_name}_best.keras")
        self.figures_dir = settings['paths']['figures_save']

    def train(self):
        self.logger.info("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN...")

        provider = DataProvider(self.settings)
        X_train, y_train, X_test, y_test = provider.load_and_split()

        provider.save_scalers()

        n_features_price = X_train['input_price'].shape[2]
        n_features_macro = X_train['input_macro'].shape[1]

        self.logger.info(f"📊 Input Features: Price={n_features_price}, Macro={n_features_macro}")

        builder = GoldPriceModel(self.settings)

        model = builder.build_model(
            input_shape_price=(X_train['input_price'].shape[1], n_features_price),
            input_shape_macro=(n_features_macro,)
        )

        learning_rate = self.train_conf.get('learning_rate', 0.001)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={'output_min': 'mse', 'output_max': 'mse'},
            metrics={'output_min': 'mae', 'output_max': 'mae'}
        )
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

        history = model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_test, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.logger.info("HUẤN LUYỆN HOÀN TẤT!")
        self.plot_history(history)

        return self.model_save_path

    def plot_history(self, history):
        """Vẽ biểu đồ và lưu vào file ảnh thay vì chỉ show"""
        os.makedirs(self.figures_dir, exist_ok=True)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Total Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        if 'val_output_min_mae' in history.history:
            plt.plot(history.history['val_output_min_mae'], label='Min MAE')
            plt.plot(history.history['val_output_max_mae'], label='Max MAE')
        else:
            pass

        plt.title('Validation MAE')
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(self.figures_dir, "training_history.png")
        plt.savefig(save_path)
        self.logger.info(f"Đã lưu biểu đồ training tại: {save_path}")
        plt.close()
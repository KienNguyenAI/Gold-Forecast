import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import logging
import matplotlib.pyplot as plt
from typing import Dict
from .data_provider import DataProvider
from src.models.hybrid_model import GoldPriceModel


class ModelTrainer:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.models_cfg = settings.get('models_config')
        self.save_dir = settings['paths']['model_save']
        self.figures_dir = settings['paths']['figures_save']

        # Lấy thông tin training chung
        self.train_conf = settings['training']

    def train(self):
        self.logger.info("🚀 BẮT ĐẦU HUẤN LUYỆN 2 MODEL (SHORT & LONG)...")

        # Vòng lặp train từng model
        for key, config in self.models_cfg.items():
            self._train_single_model(key, config)

    def _train_single_model(self, key, config):
        self.logger.info(f"\n>>> 🏋️ TRAINING: {key.upper()} ({config['name']})")

        # 1. Cập nhật settings tạm thời để DataProvider biết dùng window_size nào
        self.settings['processing']['window_size'] = config['window_size']
        self.settings['model'] = config  # Override config model chung

        # 2. Load Data & Scaler
        provider = DataProvider(self.settings)
        X_train, y_train, X_test, y_test = provider.load_and_split()

        # Lưu scaler với suffix (VD: _short_term)
        provider.save_scalers(suffix=f"_{key}")

        # 3. Build Model
        n_features_price = X_train['input_price'].shape[2]
        n_features_macro = X_train['input_macro'].shape[1]

        builder = GoldPriceModel(self.settings)
        model = builder.build_model(
            input_shape_price=(config['window_size'], n_features_price),
            input_shape_macro=(n_features_macro,)
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss={'output_min': 'mse', 'output_max': 'mse'},
            metrics={'output_min': 'mae', 'output_max': 'mae'}
        )

        save_path = os.path.join(self.save_dir, f"{config['name']}.keras")

        callbacks = [
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=self.train_conf['patience'], restore_best_weights=True)
        ]

        history = model.fit(
            X_train, y_train, validation_data=(X_test, y_test),
            epochs=self.train_conf['epochs'],
            batch_size=self.train_conf['batch_size'],
            callbacks=callbacks, verbose=1
        )

        self.plot_history(history, config['name'])
        return save_path

    def plot_history(self, history, model_name):
        os.makedirs(self.figures_dir, exist_ok=True)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f'Loss: {model_name}')
        plt.legend()

        plt.subplot(1, 2, 2)
        if 'val_output_min_mae' in history.history:
            plt.plot(history.history['val_output_min_mae'], label='Min MAE')
            plt.plot(history.history['val_output_max_mae'], label='Max MAE')

        plt.title(f'MAE: {model_name}')
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(self.figures_dir, f"history_{model_name}.png"))
        plt.close()
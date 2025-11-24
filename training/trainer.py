import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import matplotlib.pyplot as plt

# Import các class chúng ta đã viết
# Lưu ý: Python hiểu đường dẫn từ thư mục gốc dự án khi chạy main_train.py
from training.data_provider import DataProvider
from models.hybrid_model import GoldPriceModel


class GoldTrainer:
    def __init__(self, epochs=50, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_save_path = "models/best_gold_model.keras"  # Đuôi .keras là chuẩn mới của TensorFlow

        # Đảm bảo thư mục models tồn tại
        os.makedirs("models", exist_ok=True)

    def train(self):
        print("🚀 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN...")

        # 1. Chuẩn bị dữ liệu
        provider = DataProvider(window_size=60)
        X_train, y_train, X_test, y_test = provider.load_and_split()

        # Lưu lại Scaler để sau này dùng cho dự đoán thực tế
        provider.save_scalers(path="models/")

        # 2. Xây dựng mô hình
        # Lấy số lượng features động từ dữ liệu đầu vào
        n_features_price = X_train['input_price'].shape[2]  # Sẽ là 5
        n_features_macro = X_train['input_macro'].shape[1]  # Sẽ là 4

        print(f"📊 Cấu hình Input: Price Features={n_features_price}, Macro Features={n_features_macro}")

        model_builder = GoldPriceModel(
            window_size=60,
            n_features_price=n_features_price,
            n_features_macro=n_features_macro
        )
        model = model_builder.build_model()

        # 3. Compile Mô hình
        # Loss function: MSE (Mean Squared Error) để tối ưu hóa sai số bình phương
        # Metrics: MAE (Mean Absolute Error) để dễ đọc sai số thực tế
        model.compile(
            optimizer='adam',
            loss={'output_min': 'mse', 'output_max': 'mse'},
            metrics={'output_min': 'mae', 'output_max': 'mae'}
        )

        # 4. Cấu hình Callbacks (Trợ lý huấn luyện)
        callbacks = [
            # Chỉ lưu model nếu validation loss giảm (Model tốt lên)
            ModelCheckpoint(
                self.model_save_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            # Dừng train sớm nếu 10 epochs liên tiếp không tiến bộ (tránh tốn điện)
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]

        # 5. BẮT ĐẦU TRAIN (FIT)
        history = model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_test, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print("✅ HUẤN LUYỆN HOÀN TẤT!")
        self.plot_history(history)

    def plot_history(self, history):
        """Vẽ biểu đồ Loss để xem AI học thế nào"""
        plt.figure(figsize=(12, 5))

        # Vẽ Loss tổng
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Tổng Sai Số (Total Loss)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Vẽ MAE của Min và Max
        plt.subplot(1, 2, 2)
        plt.plot(history.history['val_output_min_mae'], label='Sai số Min (Val)')
        plt.plot(history.history['val_output_max_mae'], label='Sai số Max (Val)')
        plt.title('Sai số Tuyệt đối (MAE) trên tập Test')
        plt.xlabel('Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()
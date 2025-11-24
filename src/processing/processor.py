import os
import pandas as pd
import logging
from typing import Dict
from .data_merger import DataMerger
from .feature_builder import FeatureBuilder


class DataProcessor:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.processed_path = settings['paths']['processed_data']

        # Khởi tạo các worker
        self.merger = DataMerger(settings)
        self.builder = FeatureBuilder()

    def run(self):
        """Chạy toàn bộ quy trình xử lý dữ liệu"""
        self.logger.info("⚙️ --- BẮT ĐẦU QUY TRÌNH XỬ LÝ DỮ LIỆU ---")

        try:
            # 1. Ghép dữ liệu
            df = self.merger.load_and_merge()

            # 2. Tạo chỉ báo kỹ thuật
            df = self.builder.add_technical_indicators(df)

            # 3. Tạo Target (Lấy window_size từ config nếu có, mặc định 30)
            pred_window = 30  # Bạn có thể thêm vào settings.yaml
            df = self.builder.create_targets(df, prediction_window=pred_window)

            # 4. Lưu file kết quả
            os.makedirs(self.processed_path, exist_ok=True)
            save_file = os.path.join(self.processed_path, "gold_processed_features.csv")

            df.to_csv(save_file)
            self.logger.info(f"💾 Đã lưu dữ liệu sạch tại: {save_file}")
            self.logger.info(f"📊 Kích thước cuối cùng: {df.shape}")

            return save_file

        except Exception as e:
            self.logger.error(f"❌ Lỗi trong quá trình xử lý: {e}", exc_info=True)
            raise
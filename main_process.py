import os
from processor.merger import DataMerger
from processor.feature_builder import FeatureBuilder


def main():
    # 1. Khởi tạo
    merger = DataMerger()
    feature_builder = FeatureBuilder()

    # 2. Ghép dữ liệu Raw
    df_merged = merger.load_and_merge()
    print(f"✅ Đã ghép xong. Kích thước: {df_merged.shape}")

    # 3. Tạo Feature kỹ thuật (RSI, Volatility...)
    df_with_features = feature_builder.add_technical_indicators(df_merged)

    # 4. Tạo Target (Dự đoán 22 ngày trading ~ 1 tháng dương lịch)
    FINAL_DF = merger.create_targets(df_with_features, prediction_window=22)

    # 5. Lưu kết quả cuối cùng
    output_path = "data/processed/Master_Dataset.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    FINAL_DF.to_csv(output_path)

    print("-" * 30)
    print(f"✅ HOÀN TẤT! Dữ liệu sạch đã lưu tại: {output_path}")
    print("Dữ liệu này đã sẵn sàng để train AI.")
    print("Mẫu dữ liệu 5 dòng cuối:")
    print(FINAL_DF[['Gold_Close', 'Target_Min_Change', 'Target_Max_Change']].tail())


if __name__ == "__main__":
    main()
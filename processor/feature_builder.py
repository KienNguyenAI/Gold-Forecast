import pandas as pd
import numpy as np


class FeatureBuilder:
    def add_technical_indicators(self, df, price_col='Gold_Close'):
        """
        Thêm các chỉ báo kỹ thuật.
        Lưu ý: Mặc định sử dụng cột 'Gold_Close' thay vì 'Close'
        """
        df = df.copy()

        # Kiểm tra xem cột giá có tồn tại không
        if price_col not in df.columns:
            raise KeyError(f"Lỗi: Không tìm thấy cột '{price_col}' trong dữ liệu. Các cột hiện có: {list(df.columns)}")

        print(f"🛠️ Đang tạo chỉ báo kỹ thuật dựa trên cột: {price_col}")

        # 1. Log Returns (Lợi nhuận logarit)
        df['Log_Return'] = np.log(df[price_col] / df[price_col].shift(1))

        # 2. Biến động (Volatility) trong 20 ngày qua
        df['Volatility_20d'] = df['Log_Return'].rolling(window=20).std()

        # 3. RSI (Relative Strength Index)
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

        # Tránh lỗi chia cho 0
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)  # Fill NaN bằng 50 (trung tính)

        # 4. SMA Ratio (Giá hiện tại / SMA 50)
        df['SMA_50'] = df[price_col].rolling(window=50).mean()
        df['Trend_Signal'] = df[price_col] / df['SMA_50']

        return df.dropna()
import pandas as pd
import numpy as np
import logging


class FeatureBuilder:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def add_technical_indicators(self, df: pd.DataFrame, price_col='Gold_Close') -> pd.DataFrame:
        """Thêm các chỉ báo kỹ thuật (RSI, SMA, Volatility...)"""
        self.logger.info("[Step 2] Đang tính toán chỉ báo kỹ thuật...")

        if price_col not in df.columns:
            raise KeyError(f"Không tìm thấy cột '{price_col}'")

        df = df.copy()

        # 1. Log Returns
        df['Log_Return'] = np.log(df[price_col] / df[price_col].shift(1))

        # 2. Volatility 20d
        df['Volatility_20d'] = df['Log_Return'].rolling(window=20).std()

        # 3. RSI
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)

        # 4. Trend Signal (Price / SMA50)
        df['SMA_50'] = df[price_col].rolling(window=50).mean()
        df['Trend_Signal'] = df[price_col] / df['SMA_50']

        return df.dropna()

    def create_targets(self, df: pd.DataFrame, prediction_window=30) -> pd.DataFrame:
        """Tạo nhãn dự báo (Target) cho tương lai"""
        self.logger.info(f"[Step 3] Đang tạo nhãn dự báo ({prediction_window} ngày tới)...")

        # Forward Window
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=prediction_window)

        # Tính Min/Max trong 30 ngày tới
        future_min = df['Gold_Close'].rolling(window=indexer).min()
        future_max = df['Gold_Close'].rolling(window=indexer).max()

        # Tính % thay đổi so với giá hiện tại
        df['Target_Min_Change'] = (future_min - df['Gold_Close']) / df['Gold_Close']
        df['Target_Max_Change'] = (future_max - df['Gold_Close']) / df['Gold_Close']

        # Thêm nhãn xu hướng (Classification): 1 nếu tăng > 2%, 0 nếu không
        # df['Target_Direction'] = (df['Target_Max_Change'] > 0.02).astype(int)

        return df.dropna()
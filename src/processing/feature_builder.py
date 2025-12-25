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

    def create_targets(self, df: pd.DataFrame, horizons=[5, 15, 30, 126]) -> pd.DataFrame:
        self.logger.info(f"[Step 3] Đang tạo nhãn dự báo cho các mốc: {horizons}...")

        for h in horizons:
            # Forward Window riêng cho từng horizon
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=h)

            future_min = df['Gold_Close'].rolling(window=indexer).min()
            future_max = df['Gold_Close'].rolling(window=indexer).max()

            # Đặt tên cột kèm theo số ngày (VD: Target_Min_Change_126)
            df[f'Target_Min_Change_{h}'] = (future_min - df['Gold_Close']) / df['Gold_Close']
            df[f'Target_Max_Change_{h}'] = (future_max - df['Gold_Close']) / df['Gold_Close']

        return df
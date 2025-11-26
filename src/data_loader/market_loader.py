import yfinance as yf
import pandas as pd
import os
import logging
from typing import Dict


class MarketLoader:
    def __init__(self, settings: Dict):
        """
        Khởi tạo MarketLoader với cấu hình settings.
        """
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.raw_data_path = settings['paths']['raw_data']

        self.tickers = settings['data'].get('market_tickers', {
            'Gold': 'GC=F',
            'DXY': 'DX-Y.NYB',
            'SP500': '^GSPC',
            'Oil': 'CL=F'
        })

    def fetch_data(self, start_date: str = "2000-01-01"):
        self.logger.info(f"Đang tải dữ liệu thị trường từ {start_date}...")

        try:
            data = yf.download(list(self.tickers.values()), start=start_date, group_by='ticker', progress=False)

            saved_files = {}
            os.makedirs(self.raw_data_path, exist_ok=True)

            for name, ticker in self.tickers.items():
                try:
                    if len(self.tickers) > 1:
                        df = data[ticker].copy()
                    else:
                        df = data.copy()

                    if df.empty:
                        self.logger.warning(f"Không có dữ liệu cho {name} ({ticker})")
                        continue

                    df = df.dropna()
                    save_path = os.path.join(self.raw_data_path, f"{name}_daily.csv")
                    df.to_csv(save_path)

                    saved_files[name] = save_path
                    self.logger.info(f"[Market] Đã lưu: {name} -> {save_path}")
                except KeyError:
                    self.logger.warning(f"Lỗi truy cập dữ liệu cho {name} ({ticker})")

            return saved_files

        except Exception as e:
            self.logger.error(f"Lỗi nghiêm trọng khi tải Market Data: {e}")
            raise
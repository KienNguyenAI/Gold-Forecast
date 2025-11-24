import yfinance as yf
import pandas as pd
import os


class MarketLoader:
    def __init__(self):
        self.tickers = {
            'Gold': 'GC=F',
            'DXY': 'DX-Y.NYB',
            'SP500': '^GSPC',
            'Oil': 'CL=F'
        }

    def fetch_data(self, start_date="2000-01-01"):
        print(f"Đang tải dữ liệu thị trường từ {start_date}...")
        data = yf.download(list(self.tickers.values()), start=start_date, group_by='ticker')


        saved_files = {}
        for name, ticker in self.tickers.items():
            df = data[ticker].copy()
            df = df.dropna()

            # Lưu ra file CSV
            save_path = f"data/raw/{name}_daily.csv"
            # Tạo thư mục nếu chưa có
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path)
            saved_files[name] = save_path
            print(f"   ✅ Đã tải xong: {name} -> {save_path}")

        return saved_files
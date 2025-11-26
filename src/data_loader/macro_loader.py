from fredapi import Fred
import pandas as pd
import os
import logging
from typing import Dict
from dotenv import load_dotenv


class MacroLoader:
    def __init__(self, settings: Dict):
        """
        Ưu tiên lấy API Key từ file .env để bảo mật.
        """
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.raw_data_path = settings['paths']['raw_data']

        load_dotenv()

        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            api_key = settings['data'].get('fred_api_key')

        if not api_key or api_key == "YOUR_FRED_API_KEY_HERE":
            self.logger.warning("Chưa cấu hình FRED API Key chuẩn. Dữ liệu vĩ mô có thể không tải được!")
        else:
            pass

        self.fred = Fred(api_key=api_key)

        self.indicators = settings['data'].get('macro_indicators', {
            'CPI': 'CPIAUCSL',
            'Fed_Rate': 'FEDFUNDS',
            'US10Y': 'DGS10',
            'M2': 'M2SL'
        })

    def fetch_data(self, start_date: str = "2000-01-01"):
        self.logger.info(f"Đang tải dữ liệu Vĩ mô từ FRED...")
        saved_files = {}

        os.makedirs(self.raw_data_path, exist_ok=True)

        for name, series_id in self.indicators.items():
            try:
                series = self.fred.get_series(series_id, observation_start=start_date)

                df = pd.DataFrame(series, columns=['Value'])
                df.index.name = 'Date'

                df = df.ffill()

                save_path = os.path.join(self.raw_data_path, f"{name}_macro.csv")
                df.to_csv(save_path)

                saved_files[name] = save_path
                self.logger.info(f"[Macro] Đã lưu: {name}")

            except Exception as e:
                self.logger.error(f"[Macro] Lỗi tải {name} ({series_id}): {str(e)}")

        return saved_files
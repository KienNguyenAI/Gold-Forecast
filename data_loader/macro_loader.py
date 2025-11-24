from fredapi import Fred
import pandas as pd
import os


class MacroLoader:
    def __init__(self, api_key):
        self.fred = Fred(api_key=api_key)
        self.indicators = {
            'CPI': 'CPIAUCSL',  # Lạm phát Mỹ
            'FED_Funds_Rate': 'FEDFUNDS',  # Lãi suất điều hành
            'US10Y': 'DGS10',  # Lợi suất trái phiếu 10 năm
            'M2_Supply': 'M2SL',  # Cung tiền M2
            'Real_Interest_Rate': 'REAINTRATREARAT10Y'  # Lãi suất thực (Cực quan trọng)
        }

    def fetch_data(self, start_date="2000-01-01"):
        print(f"🔄 Đang tải dữ liệu Vĩ mô từ FRED...")

        # Vì dữ liệu FRED mỗi cái một khung thời gian khác nhau
        # Chúng ta sẽ tải từng cái và lưu riêng

        saved_files = {}
        for name, series_id in self.indicators.items():
            try:
                # Lấy dữ liệu
                series = self.fred.get_series(series_id, observation_start=start_date)

                # Chuyển thành DataFrame
                df = pd.DataFrame(series, columns=['Value'])
                df.index.name = 'Date'

                # Lưu file
                save_path = f"data/raw/{name}_macro.csv"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                df.to_csv(save_path)

                saved_files[name] = save_path
                print(f"   ✅ Đã tải xong: {name}")

            except Exception as e:
                print(f"   ❌ Lỗi tải {name}: {str(e)}")

        return saved_files
import pandas as pd
import os
import logging
from typing import Dict


class DataMerger:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.raw_dir = config['paths']['raw_data']

        # Quy tắc MacroLoader: {Key}_macro.csv
        self.macro_files = {
            'DXY': 'DXY_daily.csv',
            'US10Y': 'US10Y_macro.csv',
            'CPI': 'CPI_macro.csv',
            'Real_Rate': 'Real_Rate_macro.csv',
            # 'Fed_Rate': 'Fed_Rate_macro.csv',
            # 'M2': 'M2_macro.csv'
        }

    def load_and_merge(self) -> pd.DataFrame:
        self.logger.info("[Step 1] Đang ghép dữ liệu Market & Macro...")

        # --- BƯỚC 1: Load Dữ liệu Vàng ---
        gold_path = os.path.join(self.raw_dir, "Gold_daily.csv")
        if not os.path.exists(gold_path):
            raise FileNotFoundError(f"Thiếu file Gold tại {gold_path}. Hãy chạy mode 'fetch' trước!")

        df_gold = pd.read_csv(gold_path, index_col=0, parse_dates=True)

        # Chuẩn hóa tên cột
        col_map = {'Close': 'Gold_Close', 'Adj Close': 'Gold_Close', 'Volume': 'Gold_Volume'}
        df_gold = df_gold.rename(columns=col_map)

        # Chỉ giữ lại cột cần thiết
        if 'Gold_Close' in df_gold.columns:
            cols_to_keep = ['Gold_Close']
            if 'Gold_Volume' in df_gold.columns:
                cols_to_keep.append('Gold_Volume')
            df_gold = df_gold[cols_to_keep]

        # --- BƯỚC 2: Ghép Vĩ mô ---
        for name, filename in self.macro_files.items():
            path = os.path.join(self.raw_dir, filename)
            if os.path.exists(path):
                df_macro = pd.read_csv(path, index_col=0, parse_dates=True)

                # Lấy cột giá trị (thường là cột đầu tiên)
                if 'Close' in df_macro.columns:
                    target_col = 'Close'
                elif 'Adj Close' in df_macro.columns:
                    target_col = 'Adj Close'
                else:
                    target_col = df_macro.columns[0]

                df_macro = df_macro[[target_col]].rename(columns={target_col: name})

                # Left Join & Forward Fill
                df_gold = df_gold.join(df_macro, how='left')
                df_gold[name] = df_gold[name].ffill()
            else:
                self.logger.warning(f"Không tìm thấy file {filename}, dữ liệu {name} sẽ thiếu!")

        # Xóa NaN đầu dòng
        original_len = len(df_gold)
        df_gold.dropna(inplace=True)
        self.logger.info(
            f"Đã ghép xong. Dữ liệu: {len(df_gold)} dòng (Loại bỏ {original_len - len(df_gold)} dòng NaN).")

        return df_gold
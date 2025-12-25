import os
import logging
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, List
from datetime import timedelta


class GoldPredictor:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.models_cfg = settings['models_config']
        self.model_dir = settings['paths']['model_save']
        self.final_dir = settings['paths']['final_data']
        self.data_path = os.path.join(settings['paths']['processed_data'], "gold_processed_features.csv")

    def predict(self):
        self.logger.info("🚀 BẮT ĐẦU DỰ BÁO: 30 NGÀY -> 3 THÁNG -> 6 THÁNG -> 1 NĂM...")
        os.makedirs(self.final_dir, exist_ok=True)

        if not os.path.exists(self.data_path):
            self.logger.error(f"❌ Không tìm thấy dữ liệu tại: {self.data_path}")
            return

        original_df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)

        self.logger.info("1️⃣ [PHASE A] Đang tạo dữ liệu 30 ngày chuẩn...")
        df_5 = self._predict_separate('short_term', original_df, 5)
        df_15 = self._predict_separate('medium_term', original_df, 15)
        df_30 = self._predict_separate('long_term', original_df, 30)

        rows_30 = self._create_base_30_days(df_5, df_15, df_30)
        self._finalize_and_save(rows_30, "30days_final.csv")

        self.logger.info("2️⃣ [PHASE B] Đang tạo dữ liệu 3 Tháng & 6 Tháng (Round 1)...")
        df_126_r1 = self._predict_separate('semi_annual', original_df, 126)

        rows_6m = []
        if not df_126_r1.empty:
            # Tạo data 6 tháng từ phương pháp nối chuỗi
            rows_6m = self._chain_relative(base_rows=rows_30, trend_df=df_126_r1)

            # --- MỚI: TẠO FILE 3 THÁNG TỪ DATA 6 THÁNG ---
            # 3 tháng giao dịch ~ 63-65 ngày. Lấy cắt từ rows_6m
            rows_3m = rows_6m[:65]
            self._finalize_and_save(rows_3m, "3months_final.csv")

            # Lưu file 6 tháng
            self._finalize_and_save(rows_6m, "6months_final.csv")
        else:
            self.logger.warning("⚠️ Không chạy được model semi_annual. Dừng Phase B.")
            return "Error Phase B"

        self.logger.info("3️⃣ [PHASE C] Đang tạo dữ liệu 1 năm (Round 2 - Recursive)...")
        updated_df = self._update_history_data(original_df, rows_6m)
        df_126_r2 = self._predict_separate('semi_annual', updated_df, 126)

        if not df_126_r2.empty:
            rows_1y = self._chain_relative(base_rows=rows_6m, trend_df=df_126_r2)
            self._finalize_and_save(rows_1y, "1year_final.csv")
        else:
            self.logger.warning("⚠️ Không chạy được model semi_annual lần 2. Dừng Phase C.")

        return "Done"

    def _create_base_30_days(self, df_5, df_15, df_30) -> List[Dict]:
        final_rows = []
        if not df_5.empty:
            for i in range(len(df_5)): final_rows.append(df_5.iloc[i].to_dict())
        if not df_15.empty and len(df_15) >= 15:
            short_end = df_5.iloc[-1]
            medium_at_5 = df_15.iloc[4]
            gap_1 = {k: short_end[f'Forecast_{k}'] - medium_at_5[f'Forecast_{k}'] for k in ['Close', 'Min', 'Max']}
            for i in range(5, 15):
                decay = max(0, (25.0 - (i - 4)) / 25.0)
                final_rows.append(self._apply_gap(df_15.iloc[i], gap_1, decay))
        if not df_30.empty and len(df_30) >= 30:
            phase2_end = final_rows[-1]
            long_at_15 = df_30.iloc[14]
            gap_2 = {k: phase2_end[f'Forecast_{k}'] - long_at_15[f'Forecast_{k}'] for k in ['Close', 'Min', 'Max']}
            for i in range(15, 30):
                decay = max(0, (15.0 - (i - 14)) / 15.0)
                final_rows.append(self._apply_gap(df_30.iloc[i], gap_2, decay))
        return final_rows

    def _chain_relative(self, base_rows: List[Dict], trend_df: pd.DataFrame) -> List[Dict]:
        extended_rows = [row.copy() for row in base_rows]
        if not extended_rows or trend_df.empty: return extended_rows

        last_date_in_base = pd.to_datetime(extended_rows[-1]['Date'])
        last_close = extended_rows[-1]['Forecast_Close']

        trend_df = trend_df.copy()
        trend_df['Pct_Change'] = trend_df['Forecast_Close'].pct_change().fillna(0)

        to_append = trend_df[pd.to_datetime(trend_df['Date']) > last_date_in_base]

        for _, row in to_append.iterrows():
            pct = row['Pct_Change']
            new_close = last_close * (1 + pct)
            spread_min_pct = (row['Forecast_Min'] - row['Forecast_Close']) / row['Forecast_Close']
            spread_max_pct = (row['Forecast_Max'] - row['Forecast_Close']) / row['Forecast_Close']

            extended_rows.append({
                'Date': row['Date'],
                'Forecast_Close': new_close,
                'Forecast_Min': new_close * (1 + spread_min_pct),
                'Forecast_Max': new_close * (1 + spread_max_pct)
            })
            last_close = new_close

        return extended_rows

    def _update_history_data(self, original_df, projected_rows):
        new_chunk = pd.DataFrame(projected_rows)
        new_chunk = new_chunk.rename(columns={'Forecast_Close': 'Gold_Close'})
        new_chunk['Date'] = pd.to_datetime(new_chunk['Date'])
        new_chunk.set_index('Date', inplace=True)

        append_df = new_chunk[['Gold_Close']].copy()

        last_macro = original_df[['DXY', 'US10Y', 'CPI', 'Real_Rate']].iloc[-1]
        for col in ['DXY', 'US10Y', 'CPI', 'Real_Rate']:
            append_df[col] = last_macro[col]

        updated_df = pd.concat([original_df, append_df])

        updated_df['Log_Return'] = np.log(updated_df['Gold_Close'] / updated_df['Gold_Close'].shift(1))
        updated_df['Volatility_20d'] = updated_df['Log_Return'].rolling(window=20).std()
        updated_df['SMA_50'] = updated_df['Gold_Close'].rolling(window=50).mean()
        updated_df['Trend_Signal'] = updated_df['Gold_Close'] / updated_df['SMA_50']

        delta = updated_df['Gold_Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        updated_df['RSI'] = 100 - (100 / (1 + rs))

        return updated_df.ffill().bfill()

    def _finalize_and_save(self, rows: List[Dict], filename: str):
        if not rows: return
        df = pd.DataFrame(rows)

        smooth_base = df['Forecast_Close'].rolling(window=3, center=True, min_periods=1).mean()
        deviation = df['Forecast_Close'] - smooth_base
        new_close = smooth_base + (deviation * 1.5)

        spread_min = df['Forecast_Close'] - df['Forecast_Min']
        spread_max = df['Forecast_Max'] - df['Forecast_Close']

        df['Forecast_Close'] = new_close
        df['Forecast_Min'] = new_close - spread_min
        df['Forecast_Max'] = new_close + spread_max

        save_path = os.path.join(self.final_dir, filename)
        df.to_csv(save_path, index=False)
        self.logger.info(f"✅ Đã lưu file: {filename} ({len(df)} dòng)")

    def _apply_gap(self, row, gap, decay):
        return {
            'Date': row['Date'],
            'Forecast_Close': row['Forecast_Close'] + (gap['Close'] * decay),
            'Forecast_Min': row['Forecast_Min'] + (gap['Min'] * decay),
            'Forecast_Max': row['Forecast_Max'] + (gap['Max'] * decay)
        }

    def _predict_separate(self, key, df, days_to_predict):
        if key not in self.models_cfg: return pd.DataFrame()
        config = self.models_cfg[key]
        model_path = os.path.join(self.model_dir, f"{config['name']}.keras")
        if not os.path.exists(model_path): return pd.DataFrame()

        try:
            model = tf.keras.models.load_model(model_path)
            scaler_tech = joblib.load(os.path.join(self.model_dir, f"scaler_tech_{key}.pkl"))
            scaler_macro = joblib.load(os.path.join(self.model_dir, f"scaler_macro_{key}.pkl"))
        except:
            return pd.DataFrame()

        results = []
        tech_cols = ['Gold_Close', 'Log_Return', 'RSI', 'Volatility_20d', 'Trend_Signal']
        macro_cols = ['DXY', 'US10Y', 'CPI', 'Real_Rate']

        last_real_date = df.index[-1]
        current_price_T = df['Gold_Close'].iloc[-1]
        horizon = config['forecast_horizon']
        window_size = config['window_size']

        for i in range(1, days_to_predict + 1):
            target_date = last_real_date + timedelta(days=i)
            offset = max(0, horizon - i)
            if offset == 0:
                sub_df = df.iloc[-window_size:]
            else:
                sub_df = df.iloc[-(window_size + offset): -offset]

            ref_price = sub_df['Gold_Close'].iloc[-1]
            tech_s = scaler_tech.transform(sub_df[tech_cols])
            macro_s = scaler_macro.transform(sub_df[macro_cols].iloc[[-1]])

            pred = model.predict([np.expand_dims(tech_s, axis=0), macro_s], verbose=0)
            p_min, p_max = pred[0][0][0], pred[1][0][0]

            f_min = ref_price * (1 + p_min)
            f_max = ref_price * (1 + p_max)
            f_close = (f_min + f_max) / 2

            if key == 'short_term':
                past_idx = -6 + (i - 1)
                if abs(past_idx) <= len(df):
                    past_price = df['Gold_Close'].iloc[past_idx]
                    pct = (f_close - past_price) / past_price
                    f_close = current_price_T * (1 + pct)
                    f_min = f_close * (1 + (p_min - p_max) / 2)
                    f_max = f_close * (1 + (p_max - p_min) / 2)

            results.append(
                {'Date': target_date, 'Forecast_Close': f_close, 'Forecast_Min': f_min, 'Forecast_Max': f_max})

        return pd.DataFrame(results)
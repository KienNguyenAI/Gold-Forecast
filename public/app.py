import streamlit as st
import sys
import os
import pandas as pd
from datetime import timedelta

# Setup Path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import Components
from style import apply_custom_style
from components.metrics import get_stats_dictionary
from components.charts import draw_main_chart, draw_forecast_chart
from components.controls import filter_data_by_range, render_time_range_buttons
from components.header import render_market_header, render_forecast_header
from components.table import render_html_table

# Config trang (Layout Wide)
st.set_page_config(layout="wide", page_title="Gold TradingView", page_icon="📈")


# --- 1. LOAD DATA LỊCH SỬ ---
@st.cache_data
def load_history_data():
    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'gold_processed_features.csv')
    if not os.path.exists(data_path): return pd.DataFrame()
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


# --- 2. LOAD DATA DỰ BÁO ---
@st.cache_data
def load_forecast_data(filename):
    path = os.path.join(PROJECT_ROOT, 'data', 'final', filename)
    if not os.path.exists(path): return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        required_cols = ['Date', 'Forecast_Close', 'Forecast_Min', 'Forecast_Max']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        return pd.DataFrame()


# --- MARKET OVERVIEW TAB ---
def render_market_view(df):
    st.markdown("<br>", unsafe_allow_html=True)
    render_market_header(df)

    if 'time_range' not in st.session_state:
        st.session_state.time_range = '1Y'

    stats = get_stats_dictionary(df)
    render_time_range_buttons(stats)

    filtered_df = filter_data_by_range(df, st.session_state.time_range)

    st.caption(f"Hiển thị dữ liệu: {st.session_state.time_range}")
    fig = draw_main_chart(filtered_df)
    st.plotly_chart(fig, use_container_width=True)


# --- AI FORECAST TAB ---
def render_forecast_view(history_df):
    st.markdown("<br>", unsafe_allow_html=True)

    # 1. State Management (Khởi tạo mặc định)
    if 'forecast_period' not in st.session_state:
        st.session_state.forecast_period = '1 Month'

    # 2. Định nghĩa cấu hình Nút bấm (Define Config)
    buttons_config = [
        {"label": "5 Days", "key": "5 Day", "file": "5days_forecast.csv", "lookback": 30},
        {"label": "1 Month", "key": "1 Month", "file": "30days_final.csv", "lookback": 90},
        {"label": "3 Months", "key": "3 Month", "file": "3months_final.csv", "lookback": 120},
        {"label": "6 Months", "key": "6 Month", "file": "6months_final.csv", "lookback": 180},
        {"label": "1 Year", "key": "1 Year", "file": "1year_final.csv", "lookback": 460}
    ]

    # 3. Lấy cấu hình hiện tại & Load Data TRƯỚC (để vẽ Header)
    current_config = next(item for item in buttons_config if item["key"] == st.session_state.forecast_period)
    chart_df = load_forecast_data(current_config["file"])

    if chart_df.empty:
        st.warning(f"⚠️ Chưa có dữ liệu `{current_config['file']}`. Hãy chạy `python main.py predict`.")
        return

    # 4. VẼ HEADER (Title, Price, Badge) - VỊ TRÍ MỚI: TRÊN CÙNG
    render_forecast_header(history_df, chart_df)

    # Thêm khoảng cách nhỏ
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # 5. VẼ BUTTONS - VỊ TRÍ MỚI: DƯỚI HEADER
    col1, col2, col3, col4, col5, _ = st.columns([1, 1, 1, 1, 1, 3])

    for idx, col in enumerate([col1, col2, col3, col4, col5]):
        conf = buttons_config[idx]
        with col:
            btn_type = "primary" if st.session_state.forecast_period == conf["key"] else "secondary"
            if st.button(conf["label"], key=f"btn_forecast_{idx}", type=btn_type, use_container_width=True):
                st.session_state.forecast_period = conf["key"]
                st.rerun()

    # 6. Xử lý cắt Data Lịch sử & Vẽ Biểu đồ
    last_date = history_df.index[-1]
    start_date = last_date - timedelta(days=current_config["lookback"])
    sliced_history_df = history_df[history_df.index >= start_date]

    # Vẽ Chart
    fig = draw_forecast_chart(sliced_history_df, chart_df)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 7. Metrics Tóm tắt
    last_row = chart_df.iloc[-1]
    current_price = history_df['Gold_Close'].iloc[-1]
    target_price = last_row['Forecast_Close']
    change = target_price - current_price
    target_date_str = last_row['Date'].strftime('%d/%m/%Y')

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${current_price:,.2f}")

    trend_icon = "↗" if change > 0 else "↘"
    m2.metric("Predicted Trend", f"{'Increase' if change > 0 else 'Decrease'} {trend_icon}")

    m3.metric("Target Date", target_date_str)
    m4.metric("Risk Range", f"${last_row['Forecast_Min']:,.0f} - ${last_row['Forecast_Max']:,.0f}")

    # 8. Bảng chi tiết
    st.markdown("### 📋 Forecast Details (30 Days Final)")
    table_df = load_forecast_data("30days_final.csv")

    if not table_df.empty:
        latest_history_price = history_df['Gold_Close'].iloc[-1]
        render_html_table(table_df, latest_history_price)
    else:
        st.warning("⚠️ Không tìm thấy file `30days_final.csv` cho bảng chi tiết.")


# --- MAIN ---
def main():
    apply_custom_style()

    df_history = load_history_data()

    if df_history.empty:
        st.error("⚠️ Thiếu dữ liệu lịch sử! Hãy chạy pipeline trước.")
        return

    tab1, tab2 = st.tabs(["📊 Market Overview", "🔮 AI Forecast"])

    with tab1:
        render_market_view(df_history)

    with tab2:
        render_forecast_view(df_history)


if __name__ == "__main__":
    main()
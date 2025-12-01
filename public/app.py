import streamlit as st
import sys
import os
import pandas as pd

# Setup Path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import Components
from style import apply_custom_style
from components.metrics import get_stats_dictionary
from components.charts import draw_main_chart, draw_forecast_chart
from components.controls import filter_data_by_range, render_time_range_buttons

# --- THAY ĐỔI Ở ĐÂY: Import 2 hàm header mới ---
from components.header import render_market_header, render_forecast_header

# Config trang (Layout Wide)
st.set_page_config(layout="wide", page_title="Gold TradingView", page_icon="📈")


# --- HÀM LOAD DATA ---
@st.cache_data
def load_history_data():
    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'gold_processed_features.csv')
    if not os.path.exists(data_path): return pd.DataFrame()
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data
def load_forecast_data():
    path = os.path.join(PROJECT_ROOT, 'data', 'final', '30days_forecast.csv')
    if not os.path.exists(path): return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        required_cols = ['Date', 'Forecast_Close', 'Forecast_Min', 'Forecast_Max']
        if not all(col in df.columns for col in required_cols):
            st.error(f"⚠️ File dự báo thiếu cột. Hãy chạy lại dự báo.")
            return pd.DataFrame()
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Lỗi đọc file dự báo: {e}")
        return pd.DataFrame()


# --- NỘI DUNG TRANG 1: MARKET OVERVIEW ---
def render_market_view(df):
    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Header (Dùng hàm chuẩn hóa từ component)
    render_market_header(df)

    # 2. Controls & Chart
    if 'time_range' not in st.session_state:
        st.session_state.time_range = '1Y'

    stats = get_stats_dictionary(df)
    render_time_range_buttons(stats)

    filtered_df = filter_data_by_range(df, st.session_state.time_range)

    st.caption(f"Hiển thị dữ liệu: {st.session_state.time_range}")
    fig = draw_main_chart(filtered_df)

    st.plotly_chart(fig, use_container_width=True)


# --- NỘI DUNG TRANG 2: AI FORECAST ---
def render_forecast_view(history_df, forecast_df):
    st.markdown("<br>", unsafe_allow_html=True)

    if forecast_df.empty:
        st.warning("⚠️ Chưa có dữ liệu dự báo. Hãy chạy 'python main.py predict' trước.")
        return

    # 1. Header (GỌI HÀM MỚI - Code cực gọn)
    render_forecast_header(history_df, forecast_df)

    # 2. Biểu đồ (Chart)
    fig = draw_forecast_chart(history_df, forecast_df)
    st.plotly_chart(fig, use_container_width=True)

    # 3. Các chỉ số phụ (Metrics ở dưới cùng)
    # Vẫn cần tính toán một chút để hiển thị metrics dưới đáy
    last_row = forecast_df.iloc[-1]
    current_price = history_df['Gold_Close'].iloc[-1]
    target_price = last_row['Forecast_Close']
    change = target_price - current_price

    st.markdown("---")

    m1, m2, m3 = st.columns(3)

    # Metric 1
    m1.metric("Giá hiện tại (Real-time)", f"${current_price:,.2f}")

    # Metric 2
    trend_icon = "↗" if change > 0 else "↘"
    m2.metric("Xu hướng dự báo", f"{'TĂNG' if change > 0 else 'GIẢM'} {trend_icon}")

    # Metric 3
    m3.metric("Biên độ rủi ro (Min - Max)", f"${last_row['Forecast_Min']:,.0f} - ${last_row['Forecast_Max']:,.0f}")

    with st.expander("📋 Xem chi tiết dữ liệu dự báo từng ngày"):
        st.dataframe(forecast_df, width="stretch")


# --- MAIN ---
def main():
    apply_custom_style()

    df_history = load_history_data()
    df_forecast = load_forecast_data()

    if df_history.empty:
        st.error("⚠️ Thiếu dữ liệu lịch sử! Hãy chạy pipeline trước.")
        return

    tab1, tab2 = st.tabs(["📊 Market Overview", "🔮 AI Forecast"])

    with tab1:
        render_market_view(df_history)

    with tab2:
        render_forecast_view(df_history, df_forecast)


if __name__ == "__main__":
    main()
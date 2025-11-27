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
from components.header import render_header

# Config trang (Layout Wide)
st.set_page_config(layout="wide", page_title="Gold TradingView", page_icon="📈")

# --- CUSTOM CSS CHO LIGHT MODE (NỀN TRẮNG) ---
st.markdown("""
<style>
    /* 1. Ép toàn bộ App sang nền trắng */
    .stApp {
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }

    /* 2. Chỉnh màu chữ cho các thành phần chính */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #333333 !important;
    }

    /* 3. Ẩn Sidebar mặc định & Header */
    [data-testid="stSidebar"] { display: none; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }

    /* 4. Tùy chỉnh Tabs (Menu ngang) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 1px solid #E0E0E0;
        padding-bottom: 10px;
        background-color: #FFFFFF;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border: none;
        color: #666666 !important; /* Màu xám khi chưa chọn */
        font-size: 18px;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #009688 !important;
        background-color: #F5F5F5 !important;
    }

    .stTabs [aria-selected="true"] {
        color: #009688 !important; /* Màu xanh khi chọn */
        border-bottom: 3px solid #009688 !important;
    }

    /* 5. Tùy chỉnh Metrics (Các ô số liệu) */
    div[data-testid="stMetric"] {
        background-color: #F8F9FA !important; /* Nền xám cực nhạt */
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Label của Metric */
    label[data-testid="stMetricLabel"] {
        color: #666666 !important;
        font-size: 14px !important;
    }

    /* Giá trị của Metric */
    div[data-testid="stMetricValue"] {
        color: #333333 !important;
    }

    /* 6. Tùy chỉnh Expander */
    .streamlit-expanderHeader {
        background-color: #F8F9FA !important;
        color: #333333 !important;
        border-radius: 5px;
    }

    /* 7. Tùy chỉnh Table */
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


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
    st.markdown("<br>", unsafe_allow_html=True)  # Khoảng cách

    # 1. Header (Giá to)
    render_header(df)

    # 2. Controls & Chart
    if 'time_range' not in st.session_state:
        st.session_state.time_range = '1Y'

    stats = get_stats_dictionary(df)
    render_time_range_buttons(stats)  # Hàng nút bấm thời gian

    filtered_df = filter_data_by_range(df, st.session_state.time_range)

    st.caption(f"Hiển thị dữ liệu: {st.session_state.time_range}")
    fig = draw_main_chart(filtered_df)
    st.plotly_chart(fig, use_container_width=True, theme=None)


# --- NỘI DUNG TRANG 2: AI FORECAST ---
def render_forecast_view(history_df, forecast_df):
    st.markdown("<br>", unsafe_allow_html=True)

    if forecast_df.empty:
        st.warning("⚠️ Chưa có dữ liệu dự báo. Hãy chạy 'python main.py predict' trước.")
        return

    # Header dự báo
    last_row = forecast_df.iloc[-1]
    current_price = history_df['Gold_Close'].iloc[-1]
    target_price = last_row['Forecast_Close']

    change = target_price - current_price
    pct = (change / current_price) * 100

    # Hiển thị Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Giá hiện tại", f"${current_price:,.2f}")
    c2.metric("Mục tiêu 30 ngày", f"${target_price:,.2f}", f"{pct:.2f}%", delta_color="normal")
    c3.metric("Biên độ rủi ro", f"${last_row['Forecast_Min']:,.0f} - ${last_row['Forecast_Max']:,.0f}")

    st.markdown("---")

    # Chart dự báo
    st.caption("Biểu đồ dự báo xu hướng 30 ngày tới (Kèm vùng rủi ro)")
    fig = draw_forecast_chart(history_df, forecast_df)
    st.plotly_chart(fig, use_container_width=True, theme=None)

    with st.expander("📋 Xem chi tiết dữ liệu dự báo từng ngày"):
        st.dataframe(forecast_df, use_container_width=True)


# --- MAIN ---
def main():
    apply_custom_style()  # Load style chung

    # Load Data
    df_history = load_history_data()
    df_forecast = load_forecast_data()

    if df_history.empty:
        st.error("⚠️ Thiếu dữ liệu lịch sử! Hãy chạy pipeline trước.")
        return

    # 👇 TẠO NAVIGATION NGANG (DÙNG TABS THAY VÌ SIDEBAR)
    # Đây là cách tạo giao diện giống hình bạn gửi nhất
    tab1, tab2 = st.tabs(["📊 Market Overview", "🔮 AI Forecast"])

    with tab1:
        render_market_view(df_history)

    with tab2:
        render_forecast_view(df_history, df_forecast)


if __name__ == "__main__":
    main()
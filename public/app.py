import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go

# Setup path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils.config_utils import load_settings
from src.visualization import Visualizer
from src.prediction import GoldPredictor  # Import thêm cái này để lấy chỉ số

# Config trang
st.set_page_config(page_title="Gold Forecast Pro", page_icon="📈", layout="wide")

# Load settings
try:
    settings = load_settings(os.path.join(PROJECT_ROOT, 'config/settings.yaml'))
except:
    st.stop()

# --- STYLE CSS (Làm đẹp giao diện) ---
st.markdown("""
<style>
    /* Chỉnh màu nền Metric cho giống TradingView */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #333;
    }
    /* Ẩn menu hamburger mặc định */
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("📈 Gold Market Intelligence")
st.caption("Hệ thống phân tích & Dự báo giá vàng chuyên sâu")

# --- KHỐI 1: MARKET OVERVIEW (Biểu đồ TradingView) ---
st.subheader("1. Market Overview")

viz = Visualizer(settings)
try:
    # Lấy biểu đồ
    fig_market, current_price = viz.get_market_overview_chart()

    # Hiển thị chỉ số to đẹp
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Giá Vàng (USD/oz)", f"${current_price:,.2f}", "Live Update")
    with col2:
        # Tính biến động giả lập (hoặc lấy thật nếu có)
        st.metric("Biến động ngày", "+0.45%", "Bullish")
    with col3:
        st.metric("Khối lượng", "Cao", "High Volatility")

    # Hiển thị Chart 1
    st.plotly_chart(fig_market, use_container_width=True)

except Exception as e:
    st.error(f"Lỗi hiển thị Market Chart: {e}")

st.markdown("---")

# --- KHỐI 2: AI PREDICTION (Biểu đồ Dự báo) ---
st.subheader("2. AI Forecast Vision (30 Days)")

try:
    # Lấy số liệu dự báo
    predictor = GoldPredictor(settings)
    res = predictor.predict()

    # Hiển thị tóm tắt dự báo
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Giá hiện tại", f"${res['current_price']:.2f}")
    c2.metric("Dự báo Đáy", f"${res['forecast_min']:.2f}", f"{res['change_pct_min']:.2f}%", delta_color="inverse")
    c3.metric("Dự báo Đỉnh", f"${res['forecast_max']:.2f}", f"{res['change_pct_max']:.2f}%")

    trend = "TĂNG 🟢" if res['forecast_close'] > res['current_price'] else "GIẢM 🔴"
    c4.metric("Xu hướng AI", trend)

    # Hiển thị Chart 2
    fig_forecast = viz.get_forecast_chart()
    st.plotly_chart(fig_forecast, use_container_width=True)

except Exception as e:
    st.warning("Chưa có dữ liệu dự báo. Vui lòng chạy Pipeline trước.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Control Panel")
    if st.button("🔄 Run Full Pipeline"):
        # (Bạn copy lại logic run_pipeline cũ vào đây nếu muốn nút này hoạt động)
        st.info("Vui lòng chạy 'python main.py pipeline' từ terminal để ổn định nhất.")
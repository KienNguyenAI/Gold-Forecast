import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go

# --- 1. SETUP ĐƯỜNG DẪN (Kết nối với Core) ---
# Lùi lại 1 cấp thư mục để về root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils.config_utils import load_settings
from src.visualization import Visualizer
from src.prediction import GoldPredictor
from src.backtesting import Backtester
from src.training import ModelTrainer
from src.processing import DataProcessor
from src.data_loader import MarketLoader, MacroLoader

# --- 2. CONFIG TRANG WEB (Full Width, Dark Mode) ---
st.set_page_config(
    page_title="Gold Pro Terminal",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="collapsed"  # Thu gọn sidebar cho giống app trading
)

# Custom CSS để giao diện giống TradingView (Đen tuyền, Font xịn)
st.markdown("""
<style>
    .stApp { background-color: #131722; color: #d1d4dc; }

    /* Metrics box styling */
    div[data-testid="stMetric"] {
        background-color: #Q2A2E39;
        border-radius: 6px;
        padding: 10px;
        border: 1px solid #363A45;
    }
    label[data-testid="stMetricLabel"] { color: #787B86 !important; }

    /* Button styling */
    div.stButton > button {
        background-color: #2962FF;
        color: white;
        border-radius: 4px;
        border: none;
    }
    div.stButton > button:hover { background-color: #1E53E5; }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1E222D;
        border-radius: 4px 4px 0px 0px;
        color: #d1d4dc;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2962FF !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Load settings
try:
    settings = load_settings(os.path.join(PROJECT_ROOT, 'config/settings.yaml'))
except:
    st.stop()

# --- 3. HEADER & METRICS ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("GOLD / USD DOLLAR - AI PRO TERMINAL")
with col2:
    if st.button("⚡ UPDATE DATA & RUN AI"):
        with st.status("System Running...", expanded=True) as status:
            status.write("📡 Fetching Market Data...")
            MarketLoader(settings).fetch_data(settings['data']['start_date'])
            MacroLoader(settings).fetch_data(settings['data']['start_date'])
            status.write("⚙️ Processing Features...")
            DataProcessor(settings).run()
            status.write("🏋️ Retraining Model...")
            ModelTrainer(settings).train()
            status.update(label="✅ System Updated!", state="complete", expanded=False)
            st.rerun()

# --- 4. MAIN CHARTS AREA ---
viz = Visualizer(settings)

# Lấy dữ liệu dự báo
try:
    predictor = GoldPredictor(settings)
    res = predictor.predict()
    current_price = res['current_price']
except:
    current_price = 0
    res = None

# Hiển thị thanh chỉ số nhanh (Ticker Tape)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Last Price", f"${current_price:,.2f}", "+0.15%")
if res:
    m2.metric("AI Forecast Min (30d)", f"${res['forecast_min']:.2f}", f"{res['change_pct_min']:.2f}%",
              delta_color="inverse")
    m3.metric("AI Forecast Max (30d)", f"${res['forecast_max']:.2f}", f"{res['change_pct_max']:.2f}%")
    trend = "BULLISH 🚀" if res['forecast_close'] > current_price else "BEARISH 📉"
    m4.metric("Market Bias", trend)

st.markdown("---")

# TABS CHO BIỂU ĐỒ
tab_main, tab_forecast, tab_data = st.tabs(["📊 MARKET OVERVIEW", "🔮 AI VISION", "📋 RAW DATA"])

with tab_main:
    # Biểu đồ TradingView (Chart 1)
    try:
        fig_market, _ = viz.get_market_overview_chart()
        st.plotly_chart(fig_market, use_container_width=True, config={'scrollZoom': True})
    except Exception as e:
        st.error(f"Error loading Market Chart: {e}")

with tab_forecast:
    # Biểu đồ Dự báo (Chart 2)
    try:
        fig_forecast = viz.get_forecast_chart()
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.info(f"AI đang dự báo xu hướng từ ngày {res['last_date']} đến ngày {res['end_date']}")
    except Exception as e:
        st.error(f"Error loading Forecast Chart: {e}")

with tab_data:
    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'gold_processed_features.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0)
        st.dataframe(df.tail(200).style.highlight_max(axis=0), use_container_width=True)
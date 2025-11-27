import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from datetime import datetime, timedelta

# --- 1. SETUP ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

st.set_page_config(layout="wide", page_title="Gold TradingView", page_icon="📈")

# --- CSS ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .stApp { background-color: #FAFAFA; }

    /* Style chung cho nút */
    div.stButton > button {
        height: 60px;
        width: 100%;
        font-weight: bold;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        border-radius: 8px;
        transition: all 0.2s ease;
        /* Line-height để dòng chữ trên và dưới cách nhau ra một chút */
        line-height: 1.4 !important; 
    }

    /* Nút THƯỜNG (Inactive) */
    div.stButton > button[kind="secondary"] {
        background-color: #ffffff;
        border: 1px solid #ddd;
        color: #333; /* Màu chữ tiêu đề (ví dụ: "1 Month") */
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #009688;
        background-color: #Fdfdfd;
    }

    /* Nút ĐANG CHỌN (Active) */
    div.stButton > button[kind="primary"] {
        background-color: #E0F2F1 !important;
        border: 2px solid #009688 !important;
        /* Lưu ý: Không ép màu chữ ở đây bằng !important để Markdown màu xanh/đỏ hoạt động */
        color: #004D40; 
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'time_range' not in st.session_state:
    st.session_state.time_range = '1Y'


# --- 3. LOAD DATA ---
@st.cache_data
def load_data():
    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'gold_processed_features.csv')
    if not os.path.exists(data_path):
        st.error(f"⚠️ Không tìm thấy file: {data_path}")
        return pd.DataFrame()
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


def calculate_change(df, days):
    if len(df) < days: return 0.0
    current = df['Gold_Close'].iloc[-1]
    past = df['Gold_Close'].iloc[-days]
    return ((current - past) / past) * 100


def get_ytd_change(df):
    current_year = df.index[-1].year
    ytd_data = df[df.index.year == current_year]
    if ytd_data.empty: return 0.0
    start = ytd_data['Gold_Close'].iloc[0]
    current = df['Gold_Close'].iloc[-1]
    return ((current - start) / start) * 100


# --- 4. FILTER DATA ---
def filter_data(df, time_range):
    end_date = df.index.max()

    if time_range == '1M':
        start_date = end_date - timedelta(days=30)
    elif time_range == '6M':
        start_date = end_date - timedelta(days=180)
    elif time_range == 'YTD':
        start_date = datetime(end_date.year, 1, 1)
    elif time_range == '1Y':
        start_date = end_date - timedelta(days=365)
    elif time_range == '5Y':
        start_date = end_date - timedelta(days=365 * 5)
    else:
        return df

    mask = df.index >= start_date
    filtered = df.loc[mask]
    if filtered.empty: return df
    return filtered


def draw_chart(df):
    y_min = df['Gold_Close'].min()
    y_max = df['Gold_Close'].max()
    padding = (y_max - y_min) * 0.05

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Gold_Close'],
        mode='lines',
        name='Gold',
        line=dict(color='#009688', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 150, 136, 0.1)'
    ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=0, r=10, t=30, b=0),
        font=dict(color="#111"),
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False), color="#111", tickfont=dict(color="#111")),
        yaxis=dict(side='right', gridcolor='#E0E0E0', range=[y_min - padding, y_max + padding], fixedrange=False,
                   color="#111", tickfont=dict(color="#111")),
    )
    return fig


# --- 6. MAIN ---
def main():
    df = load_data()
    if df.empty: return

    stats = {
        "1M": calculate_change(df, 21),
        "6M": calculate_change(df, 126),
        "YTD": get_ytd_change(df),
        "1Y": calculate_change(df, 252),
        "5Y": calculate_change(df, 252 * 5),
        "All": ((df['Gold_Close'].iloc[-1] - df['Gold_Close'].iloc[0]) / df['Gold_Close'].iloc[0]) * 100
    }

    cols = st.columns(6)
    buttons_info = [
        ("1 Month", "1M"), ("6 Months", "6M"), ("YTD", "YTD"),
        ("1 Year", "1Y"), ("5 Years", "5Y"), ("All Time", "All")
    ]

    for col, (label, key) in zip(cols, buttons_info):
        val = stats[key]

        # 1. Xác định màu sắc dựa trên giá trị dương/âm
        # Dùng cú pháp :color[text] của Streamlit
        if val >= 0:
            color_text = "green"
            sign = "+"
        else:
            color_text = "red"
            sign = ""  # Số âm tự có dấu trừ rồi

        # 2. Tạo nội dung nút: Tên ở dòng 1, % màu ở dòng 2
        # Ví dụ kết quả: "1 Month \n :green[+5.2%]"
        btn_label = f"{label}\n:{color_text}[{sign}{val:.2f}%]"

        # 3. Xác định trạng thái Active/Inactive
        is_active = (st.session_state.time_range == key)
        btn_type = "primary" if is_active else "secondary"

        if col.button(btn_label, key=key, type=btn_type, use_container_width=True):
            st.session_state.time_range = key
            st.rerun()

    filtered_df = filter_data(df, st.session_state.time_range)
    st.caption(f"Đang hiển thị dữ liệu: {st.session_state.time_range}")
    st.plotly_chart(draw_chart(filtered_df), use_container_width=True)


if __name__ == "__main__":
    main()
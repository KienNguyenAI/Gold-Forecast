# File: ui/components/header.py
import streamlit as st


def render_header(df):
    """Hiển thị Header giống Yahoo Finance / TradingView"""
    if df.empty: return

    # Lấy giá mới nhất và giá ngày hôm qua
    current_price = df['Gold_Close'].iloc[-1]
    prev_price = df['Gold_Close'].iloc[-2]

    change = current_price - prev_price
    change_pct = (change / prev_price) * 100

    # Màu sắc (Xanh/Đỏ)
    color = "#f23645" if change < 0 else "#009688"
    sign = "" if change < 0 else "+"

    # Format ngày
    last_date = df.index[-1].strftime('%b %d, %Y')

    # Render HTML
    st.markdown(f"""
    <div style="padding-bottom: 20px;">
        <div style="font-size: 32px; font-weight: bold; color: #111; display: flex; align-items: baseline; gap: 10px;">
            Gold Price (GC=F) 
            <span style="font-size: 14px; font-weight: normal; color: #555; background: #eee; padding: 2px 8px; border-radius: 10px;">Future</span>
        </div>
        <div style="font-size: 42px; font-weight: 900; color: #111; line-height: 1.2;">
            {current_price:,.2f}
            <span style="font-size: 24px; font-weight: bold; color: {color}; margin-left: 10px;">
                {change:+.2f} ({change_pct:+.2f}%)
            </span>
        </div>
        <div style="font-size: 14px; color: #666;">
            Last updated: {last_date} • Market Data (Delayed)
        </div>
    </div>
    """, unsafe_allow_html=True)
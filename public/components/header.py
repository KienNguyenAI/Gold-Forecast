import streamlit as st

def render_html_header(title, badge_text, price, change, pct, date_text):
    """Hàm vẽ HTML Header chung"""
    if change >= 0:
        color = "#009688"
        sign = "+"
    else:
        color = "#f23645"
        sign = ""

    st.markdown(f"""
    <div style="padding-bottom: 20px;">
        <div style="font-size: 32px; font-weight: bold; color: #111; display: flex; align-items: baseline; gap: 10px;">
            {title}
            <span style="font-size: 14px; font-weight: normal; color: #555; background: #eee; padding: 2px 8px; border-radius: 10px;">
                {badge_text}
            </span>
        </div>
        <div style="font-size: 42px; font-weight: 900; color: #111; line-height: 1.2;">
            {price:,.2f}
            <span style="font-size: 24px; font-weight: bold; color: {color}; margin-left: 10px;">
                {sign}{change:,.2f} ({sign}{pct:.2f}%)
            </span>
        </div>
        <div style="font-size: 14px; color: #666;">
            {date_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_market_header(df):
    if df.empty: return
    current_price = df['Gold_Close'].iloc[-1]
    prev_price = df['Gold_Close'].iloc[-2]
    change = current_price - prev_price
    pct = (change / prev_price) * 100
    last_date = df.index[-1].strftime('%b %d, %Y')

    render_html_header("Gold Price (GC=F)", "Future • Market Data", current_price, change, pct, f"Last updated: {last_date} • Delayed")

def render_forecast_header(history_df, forecast_df):
    if forecast_df.empty: return
    last_row = forecast_df.iloc[-1]
    current_price = history_df['Gold_Close'].iloc[-1]
    target_price = last_row['Forecast_Close']
    change = target_price - current_price
    pct = (change / current_price) * 100
    target_date = last_row['Date'].strftime('%b %d, %Y')

    render_html_header("Gold Forecast", "AI Prediction (30 Days)", target_price, change, pct, f"Target Date: {target_date} • Based on Current Price: ${current_price:,.2f}")
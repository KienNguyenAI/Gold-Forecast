import streamlit as st
from datetime import datetime, timedelta


def filter_data_by_range(df, time_range):
    """Lọc DataFrame theo khoảng thời gian"""
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
        return df  # All

    mask = df.index >= start_date
    filtered = df.loc[mask]
    return filtered if not filtered.empty else df


def render_time_range_buttons(stats):
    """Hiển thị hàng nút bấm"""
    cols = st.columns(6)
    buttons_info = [
        ("1 Month", "1M"), ("6 Months", "6M"), ("YTD", "YTD"),
        ("1 Year", "1Y"), ("5 Years", "5Y"), ("All Time", "All")
    ]

    for col, (label, key) in zip(cols, buttons_info):
        val = stats[key]

        # Logic màu sắc
        if val >= 0:
            color_text = "green"
            sign = "+"
        else:
            color_text = "red"
            sign = ""

        btn_label = f"{label}\n:{color_text}[{sign}{val:.2f}%]"

        # Trạng thái nút
        is_active = (st.session_state.get('time_range', '1Y') == key)
        btn_type = "primary" if is_active else "secondary"

        if col.button(btn_label, key=key, type=btn_type, use_container_width=True):
            st.session_state.time_range = key
            st.rerun()
import streamlit as st
import pandas as pd
import math


def render_html_table(forecast_df, current_price):
    """
    Render bảng dự báo với PHÂN TRANG DẠNG NÚT BẤM (Compact Style).
    Các nút bấm sẽ nằm sát nhau hơn.
    """

    # --- 1. LOGIC TÍNH TOÁN SỐ TRANG ---
    ROWS_PER_PAGE = 5

    if 'page_number' not in st.session_state:
        st.session_state.page_number = 0

    total_rows = len(forecast_df)
    total_pages = math.ceil(total_rows / ROWS_PER_PAGE)

    # Validate page number
    if st.session_state.page_number >= total_pages:
        st.session_state.page_number = total_pages - 1
    if st.session_state.page_number < 0:
        st.session_state.page_number = 0

    # Cắt dữ liệu
    start_idx = st.session_state.page_number * ROWS_PER_PAGE
    end_idx = start_idx + ROWS_PER_PAGE
    page_df = forecast_df.iloc[start_idx:end_idx]

    # --- 2. RENDER BẢNG HTML (Giữ nguyên) ---
    styles = """
    <style>
        .custom-table { 
            width: 100%; 
            border-collapse: collapse; 
            font-family: sans-serif; 
            margin-top: 15px; 
            margin-bottom: 30px; 
        }

        .custom-table th { 
            text-align: left; padding: 20px 15px; 
            color: #000000 !important; font-weight: 800; font-size: 22px !important; 
            border-bottom: 3px solid #333; 
        }

        .custom-table td { 
            padding: 20px 15px; border-bottom: 1px solid #ddd; 
            color: #222222 !important; font-size: 20px !important; 
            vertical-align: middle; line-height: 1.5;
        }

        .price-cell { font-weight: 600; color: #000 !important; }
        .trend-icon { margin-right: 12px; font-size: 22px !important; display: inline-block; transform: translateY(2px); }
        .color-up { color: #009688 !important; }
        .color-down { color: #d32f2f !important; }
        .pct-bold { font-weight: 700; }

        /* CSS cho nút bấm: Giảm padding mặc định để nút nhỏ gọn hơn */
        div.stButton > button {
            border-radius: 4px;
            font-weight: bold;
            padding: 0px 10px !important; /* Ép nút gọn lại */
            min-height: 45px;
        }

    </style>
    """

    html_parts = []
    html_parts.append(styles)
    html_parts.append("<table class='custom-table'>")
    html_parts.append(
        "<thead><tr><th style='width: 30%'>Date</th><th style='width: 40%'>Prediction</th><th style='width: 30%'>Change (vs Now)</th></tr></thead>")
    html_parts.append("<tbody>")

    for index, row in page_df.iterrows():
        date_str = row['Date'].strftime('%b %d, %Y')
        pred_price = row['Forecast_Close']
        change_val = pred_price - current_price
        pct_change = (change_val / current_price) * 100

        if change_val >= 0:
            color_class, icon, sign = "color-up", "↑", "+"
        else:
            color_class, icon, sign = "color-down", "↓", ""

        row_str = f"<tr><td>{date_str}</td>"
        row_str += f"<td class='price-cell'><span class='{color_class} trend-icon'>{icon}</span> ${pred_price:,.2f}</td>"
        row_str += f"<td><span class='{color_class} pct-bold'>{sign}{pct_change:.2f}%</span></td></tr>"
        html_parts.append(row_str)

    html_parts.append("</tbody></table>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)

    # --- 3. THANH ĐIỀU HƯỚNG (ĐÃ SỬA ĐỂ SÁT NHAU) ---

    # Tổng số nút = [Previous] + [Các trang] + [Next]
    num_buttons = total_pages + 2

    # --- LOGIC MỚI: ÉP CỘT GIỮA ---
    # Ta dùng 3 cột: [Trống Trái] - [Cụm Nút] - [Trống Phải]
    # Để các nút sát nhau, cột [Cụm Nút] phải nhỏ.
    # Ta set tỷ lệ cột Trống thật lớn (ví dụ gấp 3 lần số nút) để ép cột giữa bé lại.

    spacer_ratio = num_buttons * 2.5  # Tỷ lệ khoảng trống lớn

    _, center_col, _ = st.columns([spacer_ratio, num_buttons, spacer_ratio])

    with center_col:
        # gap="small": Giảm khoảng cách giữa các cột con xuống mức tối thiểu
        cols = st.columns(num_buttons, gap="small")

        # 3.1. Nút Previous [«]
        if cols[0].button("«", key="prev_page", disabled=(st.session_state.page_number == 0)):
            st.session_state.page_number -= 1
            st.rerun()

        # 3.2. Các nút số
        for i in range(total_pages):
            btn_type = "primary" if st.session_state.page_number == i else "secondary"
            if cols[i + 1].button(str(i + 1), key=f"page_{i}", type=btn_type):
                st.session_state.page_number = i
                st.rerun()

        # 3.3. Nút Next [»]
        if cols[num_buttons - 1].button("»", key="next_page",
                                        disabled=(st.session_state.page_number == total_pages - 1)):
            st.session_state.page_number += 1
            st.rerun()
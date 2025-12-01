import streamlit as st


def apply_custom_style():
    st.markdown("""
    <style>
        /* --- 1. CẤU HÌNH CHUNG (LIGHT MODE) --- */
        header {visibility: hidden;}
        #MainMenu { visibility: hidden; }
        [data-testid="stSidebar"] { display: none; }

        .stApp { 
            background-color: #FFFFFF !important; 
            color: #333333 !important;
        }

        /* Chỉnh màu chữ mặc định cho các thẻ văn bản */
        h1, h2, h3, h4, h5, h6, p, div, span, label {
            color: #333333 !important;
        }

        /* --- 2. BUTTONS (NÚT BẤM) --- */
        div.stButton > button {
            height: 60px;
            width: 100%;
            font-weight: bold;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            border-radius: 8px;
            transition: all 0.2s ease;
            line-height: 1.4 !important; 
        }

        /* Nút THƯỜNG (Inactive) */
        div.stButton > button[kind="secondary"] {
            background-color: #ffffff;
            border: 1px solid #ddd;
            color: #333;
        }
        div.stButton > button[kind="secondary"]:hover {
            border-color: #009688;
            background-color: #Fdfdfd;
        }

        /* Nút ĐANG CHỌN (Active) */
        div.stButton > button[kind="primary"] {
            background-color: #E0F2F1 !important;
            border: 2px solid #009688 !important;
            color: #004D40 !important; 
        }

        /* --- 3. TABS (MENU NGANG) --- */

        /* ẨN THANH HIGHLIGHT MẶC ĐỊNH (Cái vạch đỏ) */
        div[data-baseweb="tab-highlight"] {
            display: none !important;
        }

        /* Container của các tab */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            border-bottom: 1px solid #E0E0E0;
            padding-bottom: 10px;
            background-color: #FFFFFF;
        }

        /* Style cho từng Tab (Chưa chọn) */
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border: none;
            color: #666666 !important; 
            font-size: 18px;
            font-weight: 600;
            box-shadow: none !important; /* Xóa bóng mờ nếu có */
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #009688 !important;
            background-color: #F5F5F5 !important;
        }

        /* Style cho Tab ĐANG CHỌN (Chỉ hiện vạch xanh) */
        .stTabs [aria-selected="true"] {
            color: #009688 !important; 
            background-color: transparent !important;
            border-bottom: 3px solid #009688 !important; /* Vạch xanh lá */
        }

        /* --- 4. METRICS (CÁC Ô SỐ LIỆU) --- */
        div[data-testid="stMetric"] {
            background-color: #F8F9FA !important;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #E0E0E0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        label[data-testid="stMetricLabel"] {
            color: #666666 !important;
            font-size: 14px !important;
        }

        div[data-testid="stMetricValue"] {
            color: #333333 !important;
        }

        /* --- 5. EXPANDER & TABLE --- */
        .streamlit-expanderHeader {
            background-color: #F8F9FA !important;
            color: #333333 !important;
            border-radius: 5px;
        }

        [data-testid="stDataFrame"] {
            background-color: #FFFFFF !important;
        }
    </style>
    """, unsafe_allow_html=True)
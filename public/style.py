import streamlit as st

def apply_custom_style():
    st.markdown("""
    <style>
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

        div.stButton > button[kind="primary"] {
            background-color: #E0F2F1 !important;
            border: 2px solid #009688 !important;
            color: #004D40; 
        }
    </style>
    """, unsafe_allow_html=True)
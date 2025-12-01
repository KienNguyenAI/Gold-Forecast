import plotly.graph_objects as go
import pandas as pd

# --- MÀU SẮC CHỦ ĐẠO ---
MAIN_COLOR = '#009688'  # Màu xanh Teal
FILL_COLOR = 'rgba(0, 150, 136, 0.1)'  # Màu tô mờ bên dưới
TEXT_COLOR = '#000000'  # Màu chữ đen tuyền


def draw_main_chart(df):
    if df.empty: return go.Figure()

    y_min = df['Gold_Close'].min()
    y_max = df['Gold_Close'].max()
    padding = (y_max - y_min) * 0.05

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Gold_Close'],
        mode='lines',
        name='Gold',
        line=dict(color=MAIN_COLOR, width=2),
        fill='tozeroy',
        fillcolor=FILL_COLOR
    ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=0, r=10, t=30, b=0),
        font=dict(color=TEXT_COLOR),
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False), color=TEXT_COLOR, tickfont=dict(color=TEXT_COLOR)),

        yaxis=dict(
            side='right',
            gridcolor='#E0E0E0',
            range=[y_min - padding, y_max + padding],
            fixedrange=False,
            color=TEXT_COLOR,
            tickfont=dict(color=TEXT_COLOR)
        ),
    )
    return fig


def draw_forecast_chart(history_df, forecast_df):
    """
    Vẽ biểu đồ Dự báo: Đã BỎ VÙNG RỦI RO (Chỉ còn đường dự báo chính)
    """
    fig = go.Figure()

    # 1. Lấy dữ liệu lịch sử gần nhất (90 ngày)
    recent_history = history_df.tail(90)

    # --- XỬ LÝ NỐI DỮ LIỆU (FIX GAP) ---
    plot_forecast_df = forecast_df.copy()
    if not recent_history.empty and not plot_forecast_df.empty:
        last_date = recent_history.index[-1]
        last_price = recent_history['Gold_Close'].iloc[-1]

        # Tạo cầu nối
        bridge_row = pd.DataFrame({
            'Date': [last_date],
            'Forecast_Close': [last_price],
            'Forecast_Min': [last_price],
            'Forecast_Max': [last_price]
        })

        plot_forecast_df = pd.concat([bridge_row, plot_forecast_df], ignore_index=True)
    # -----------------------------------

    # 2. TÍNH TOÁN MIN/MAX (Vẫn tính Min/Max toàn cục để trục Y hiển thị đẹp)
    min_vals = [recent_history['Gold_Close'].min()]
    max_vals = [recent_history['Gold_Close'].max()]

    if not plot_forecast_df.empty:
        # Vẫn dùng dữ liệu Min/Max để scale trục, dù không vẽ nó ra
        min_vals.append(plot_forecast_df['Forecast_Min'].min())
        max_vals.append(plot_forecast_df['Forecast_Max'].max())

    global_min = min(min_vals)
    global_max = max(max_vals)
    padding = (global_max - global_min) * 0.05
    y_range_limit = [global_min - padding, global_max + padding]

    # 3. Vẽ Lịch sử
    fig.add_trace(go.Scatter(
        x=recent_history.index,
        y=recent_history['Gold_Close'],
        mode='lines',
        name='Lịch sử',
        line=dict(color=MAIN_COLOR, width=2),
        fill='tozeroy',
        fillcolor=FILL_COLOR
    ))

    # 4. Vẽ Dự báo (ĐÃ BỎ CÁC ĐOẠN VẼ VÙNG RỦI RO)
    if not plot_forecast_df.empty:
        # --- [ĐÃ XÓA] Đoạn code vẽ Forecast_Max và Forecast_Min ở đây ---

        # Chỉ vẽ Đường Dự báo chính
        fig.add_trace(go.Scatter(
            x=plot_forecast_df['Date'],
            y=plot_forecast_df['Forecast_Close'],
            mode='lines',
            name='Dự báo AI',
            line=dict(color=MAIN_COLOR, width=2, dash='dash'),
            fill='tozeroy',  # Vẫn giữ tô màu bên dưới đường chính cho đẹp
            fillcolor=FILL_COLOR
        ))

    # 5. Cấu hình giao diện
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        margin=dict(l=0, r=10, t=10, b=0),
        hovermode="x unified",
        font=dict(family="Sans-serif", size=12, color=TEXT_COLOR),

        xaxis=dict(
            color=TEXT_COLOR,
            tickfont=dict(color=TEXT_COLOR),
            showgrid=False
        ),

        yaxis=dict(
            side='right',
            gridcolor='#E0E0E0',
            color=TEXT_COLOR,
            tickfont=dict(color=TEXT_COLOR),
            range=y_range_limit,
            fixedrange=False
        ),

        legend=dict(
            orientation="h", y=1, x=0,
            font=dict(color=TEXT_COLOR)
        )
    )

    return fig
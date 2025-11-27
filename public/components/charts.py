import plotly.graph_objects as go

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


def draw_forecast_chart(history_df, forecast_df):
    """
    Vẽ biểu đồ Dự báo (Light Mode - Đã sửa lỗi chữ mờ)
    """
    fig = go.Figure()

    # 1. Vẽ Lịch sử
    recent_history = history_df.tail(90)
    fig.add_trace(go.Scatter(
        x=recent_history.index,
        y=recent_history['Gold_Close'],
        mode='lines',
        name='Lịch sử',
        line=dict(color='#FFD700', width=2)  # Vàng đậm
    ))

    # 2. Vẽ Dự báo
    if not forecast_df.empty:
        # Đường dự báo trung bình
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast_Close'],
            mode='lines',
            name='Dự báo AI',
            line=dict(color='#00C853', width=2, dash='dash')  # Xanh lá đậm
        ))

        # Vùng an toàn
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast_Max'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast_Min'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 200, 83, 0.15)',  # Xanh lá mờ
            name='Vùng rủi ro',
            hoverinfo='skip'
        ))

    # 3. Cấu hình giao diện Light Mode
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        margin=dict(l=0, r=10, t=40, b=0),
        hovermode="x unified",

        # 👇 ĐÂY LÀ PHẦN QUAN TRỌNG VỪA THÊM VÀO
        font=dict(
            family="Sans-serif",
            size=12,
            color="#333333"  # Ép màu chữ đen xám đậm
        ),

        xaxis=dict(
            color="#333333",  # Màu chữ trục X
            showgrid=False
        ),

        yaxis=dict(
            side='right',
            gridcolor='#E0E0E0',
            color="#333333",  # Màu chữ trục Y
            autorange=True,
            fixedrange=False
        ),

        title=dict(
            text="AI Forecast Vision (Next 30 Days)",
            font=dict(size=20, color="#111111")  # Tiêu đề đen tuyền
        ),
        legend=dict(
            orientation="h", y=1, x=0,
            font=dict(color="#333333")  # Chú thích màu đen
        )
    )

    return fig
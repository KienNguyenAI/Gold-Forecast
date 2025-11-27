import pandas as pd

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

def get_stats_dictionary(df):
    return {
        "1M": calculate_change(df, 21),
        "6M": calculate_change(df, 126),
        "YTD": get_ytd_change(df),
        "1Y": calculate_change(df, 252),
        "5Y": calculate_change(df, 252 * 5),
        "All": ((df['Gold_Close'].iloc[-1] - df['Gold_Close'].iloc[0]) / df['Gold_Close'].iloc[0]) * 100
    }
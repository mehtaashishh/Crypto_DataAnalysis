import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
import requests

def get_gold_data(start_date, end_date):
    """Fetches Gold price data from Yahoo Finance."""
    try:
        gold_data = yf.download("GC=F", start=start_date, end=end_date)
        if gold_data.empty:
            print("No data fetched for the given date range.")
            return None
        return gold_data
    except Exception as e:
        print(f"Error fetching data from yfinance: {e}")
        return None

def create_interactive_gold_chart(data, start_date):
    """Creates an interactive Gold price chart with Plotly."""
    if data is None or data.empty:
        print("No data to plot.")
        return
        
    df = data.copy()
    
    # Flatten the multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.rename(columns={'Close': 'price'}, inplace=True)
    df['date'] = pd.to_datetime(df.index)
    
    start_datetime = pd.to_datetime(start_date).tz_localize(df.index.tz)
    df = df[df.index >= start_datetime]

    df = df.sort_index().drop_duplicates(keep='first')

    # Ensure 'days_since_start' is positive for log calculation
    df['days_since_start'] = (df.index - start_datetime).days
    df = df[df['days_since_start'] > 0]
    
    # Ensure we have enough data for regression
    if len(df) < 2:
        print("Not enough data points for regression analysis.")
        return

    # Filter out rows where price is zero or negative before taking the log
    df = df[df['price'] > 0]

    X = sm.add_constant(np.log(df['days_since_start']))
    y = np.log(df['price'])

    res_median = sm.QuantReg(y, X).fit(q=0.5)
    res_upper_95 = sm.QuantReg(y, X).fit(q=0.95)
    res_lower_05 = sm.QuantReg(y, X).fit(q=0.05)
    
    future_days = (datetime(2030, 1, 1) - start_datetime.replace(tzinfo=None)).days
    extended_days = np.arange(df['days_since_start'].min(), future_days)
    
    extended_days_log = extended_days[extended_days > 0]
    extended_X = sm.add_constant(np.log(extended_days_log))
    
    df_extended = pd.DataFrame({
        'date': [start_datetime.replace(tzinfo=None) + timedelta(days=int(d)) for d in extended_days_log],
        'days_since_start': extended_days_log,
        'fit': np.exp(res_median.predict(extended_X)),
        'upper_bound_95': np.exp(res_upper_95.predict(extended_X)),
        'lower_bound_05': np.exp(res_lower_05.predict(extended_X)),
    })
    
    r2 = r2_score(y, res_median.predict(X))

    fig = go.Figure()

    # Add price trace
    fig.add_trace(go.Scatter(
        x=df['days_since_start'], 
        y=df['price'], 
        mode='lines', 
        name='Price',
        customdata=df[['date', 'price']],
        hovertemplate='<b>Date</b>: %{customdata[0]|%Y-%m-%d}<br><b>Price</b>: $%{customdata[1]:,.2f}<extra></extra>'
    ))

    # Add regression lines
    for col, name, dash in [
        ('fit', f'Median Fit (R²={r2:.2f})', 'dash'),
        ('upper_bound_95', '95th Percentile', 'dot'),
        ('lower_bound_05', '5th Percentile', 'dot'),
    ]:
        fig.add_trace(go.Scatter(
            x=df_extended['days_since_start'], 
            y=df_extended[col], 
            mode='lines', 
            name=name, 
            line=dict(dash=dash),
            customdata=df_extended[['date', col]],
            hovertemplate='<b>Date</b>: %{customdata[0]|%Y-%m-%d}<br><b>Price</b>: $%{customdata[1]:,.2f}<extra></extra>'
        ))

    tick_vals = [365, 730, 1095, 1460, 1825, 2190, 2555, 2920, 3285, 3650, 4015, 4380, 4745, 5110, 5475, 5840, 6205, 6570, 6935, 7300, 7665, 8030, 8395, 8760]
    tick_text = [(start_datetime.replace(tzinfo=None) + timedelta(days=val)).strftime('%Y-%m-%d') for val in tick_vals]

    fig.update_layout(
        title='Interactive Gold Price History - Log-Log Scale with Quantile Regression',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_type='log',
        yaxis_type='log',
        xaxis=dict(
            tickvals=tick_vals,
            ticktext=tick_text,
        ),
        legend_title_text='Legend',
        hovermode='x unified'
    )
    
    # --- Create CSV with Predictions ---
    # Create a DataFrame with the predictions
    predictions_df = df_extended[['date', 'lower_bound_05', 'fit', 'upper_bound_95']].copy()
    predictions_df.rename(columns={
        'date': 'Date',
        'lower_bound_05': '5th Percentile',
        'fit': 'Median',
        'upper_bound_95': '95th Percentile'
    }, inplace=True)

    # Format the price columns as currency strings
    for col in ['5th Percentile', 'Median', '95th Percentile']:
        predictions_df[col] = predictions_df[col].apply(lambda x: f'${x:,.2f}')

    predictions_df['Date'] = pd.to_datetime(predictions_df['Date']).dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    predictions_df.to_csv('gold_price_predictions.csv', index=False)
    print("Gold price predictions saved to gold_price_predictions.csv")
    
    fig.write_html("interactive_gold_chart.html", include_plotlyjs=True)
    print("Interactive chart saved to interactive_gold_chart.html")

if __name__ == "__main__":
    start_date = "1950-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    
    gold_data = get_gold_data(start_date, end_date)
    create_interactive_gold_chart(gold_data, start_date)
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score

def get_bitcoin_data(start_date, end_date):
    """Fetches Bitcoin price data from the CryptoCompare API."""
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    
    all_data = []
    current_timestamp = end_timestamp

    while True:
        url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=2000&toTs={current_timestamp}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data['Response'] == 'Success' and 'Data' in data and 'Data' in data['Data']:
                new_data = data['Data']['Data']
                if not new_data:
                    break

                oldest_timestamp_in_chunk = new_data[0]['time']
                all_data.extend(new_data)
                current_timestamp = oldest_timestamp_in_chunk

                if oldest_timestamp_in_chunk <= start_timestamp:
                    break
            else:
                print(f"Error in API response: {data.get('Message', 'Unknown error')}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
            
    return all_data

def create_interactive_bitcoin_chart(data, start_date):
    """Creates an interactive Bitcoin price chart with Plotly."""
    if not data:
        print("No data to plot.")
        return
        
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    df = df[df['time'] >= start_datetime]

    df = df.sort_values('time').drop_duplicates(subset=['time'], keep='first')

    df['days_since_start'] = (df['time'] - start_datetime).dt.days + 1
    df = df[df['days_since_start'] >= 365]

    X = sm.add_constant(np.log(df['days_since_start']))
    y = np.log(df['close'])

    res_median = sm.QuantReg(y, X).fit(q=0.5)
    res_upper_95 = sm.QuantReg(y, X).fit(q=0.95)
    res_lower_05 = sm.QuantReg(y, X).fit(q=0.05)
    res_upper_99 = sm.QuantReg(y, X).fit(q=0.99)
    res_lower_10 = sm.QuantReg(y, X).fit(q=0.10)
    res_upper_90 = sm.QuantReg(y, X).fit(q=0.90)
    res_lower_30 = sm.QuantReg(y, X).fit(q=0.30)
    
    future_days = (datetime(2120, 1, 1) - start_datetime).days + 1
    extended_days = np.arange(df['days_since_start'].min(), future_days)
    extended_dates = [start_datetime + timedelta(days=int(d)) for d in extended_days]
    extended_X = sm.add_constant(np.log(extended_days))
    
    df_extended = pd.DataFrame({
        'date': extended_dates,
        'days_since_start': extended_days,
        'fit': np.exp(res_median.predict(extended_X)),
        'upper_bound_95': np.exp(res_upper_95.predict(extended_X)),
        'lower_bound_05': np.exp(res_lower_05.predict(extended_X)),
        'upper_bound_99': np.exp(res_upper_99.predict(extended_X))
    })
    
    r2 = r2_score(y, res_median.predict(X))

    fig = go.Figure()

    # Add price trace
    fig.add_trace(go.Scatter(
        x=df['days_since_start'], 
        y=df['close'], 
        mode='lines', 
        name='Price',
        customdata=df['time'],
        hovertemplate='<b>Date</b>: %{customdata|%Y-%m-%d}<br><b>Price</b>: $%{y:,.2f}<extra></extra>'
    ))

    # Add regression lines
    for col, name, dash in [
        ('fit', f'Median Fit (R²={r2:.2f})', 'dash'),
        ('upper_bound_95', '95th Percentile', 'dot'),
        ('lower_bound_05', '5th Percentile', 'dot'),
        ('upper_bound_99', '99th Percentile', 'dashdot')
    ]:
        fig.add_trace(go.Scatter(
            x=df_extended['days_since_start'], 
            y=df_extended[col], 
            mode='lines', 
            name=name, 
            line=dict(dash=dash),
            customdata=df_extended['date'],
            hovertemplate='<b>Date</b>: %{customdata|%Y-%m-%d}<br><b>Price</b>: $%{y:,.2f}<extra></extra>'
        ))

    halving_dates = [
        datetime(2012, 11, 28), datetime(2016, 7, 9),
        datetime(2020, 5, 11), datetime(2024, 4, 20)
    ]
    halving_days = [(d - start_datetime).days + 1 for d in halving_dates]
    for i, day in enumerate(halving_days):
        fig.add_vline(x=day, line_width=1, line_dash="dash", line_color="grey", annotation_text=f"Halving {halving_dates[i].year}", annotation_position="top left")

    tick_vals = [365, 730, 1095, 1460, 1825, 2190, 2555, 2920, 3285, 3650, 7300, 10950, 14600, 18250, 21900, 25550, 29200, 32850, 36500]
    tick_text = [(start_datetime + timedelta(days=val)).strftime('%Y-%m-%d') for val in tick_vals]

    fig.update_layout(
        title='Interactive Bitcoin Price History - Log-Log Scale with Quantile Regression',
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

    fig.write_html("interactive_bitcoin_chart.html")
    print("Interactive chart saved to interactive_bitcoin_chart.html")

    # --- Create CSV ---
    table_dates = pd.to_datetime(pd.date_range(start='2024-01-01', end='2120-01-01', freq='6MS'))
    table_days = (table_dates - start_datetime).days + 1
    table_X = sm.add_constant(np.log(table_days))
    
    table_data = {
        'Date': [d.strftime('%Y-%m-%d') for d in table_dates],
        '5th': [f'${p:,.2f}' for p in np.exp(res_lower_05.predict(table_X))],
        '10th': [f'${p:,.2f}' for p in np.exp(res_lower_10.predict(table_X))],
        '30th': [f'${p:,.2f}' for p in np.exp(res_lower_30.predict(table_X))],
        '50th': [f'${p:,.2f}' for p in np.exp(res_median.predict(table_X))],
        '90th': [f'${p:,.2f}' for p in np.exp(res_upper_90.predict(table_X))],
        '95th': [f'${p:,.2f}' for p in np.exp(res_upper_95.predict(table_X))],
        '99th': [f'${p:,.2f}' for p in np.exp(res_upper_99.predict(table_X))]
    }
    df_table = pd.DataFrame(table_data)
    df_table.to_csv('bitcoin_price_predictions.csv', index=False)
    # --- End Create CSV ---

if __name__ == "__main__":
    start_date = "2010-07-17"
    end_date = datetime.today().strftime("%Y-%m-%d")
    
    price_data = get_bitcoin_data(start_date, end_date)
    if price_data:
        create_interactive_bitcoin_chart(price_data, start_date)

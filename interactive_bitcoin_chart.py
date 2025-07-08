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

def generate_retirement_table_html(df_predictions):
    """Generates an HTML table for the retirement verification data."""
    # --- Constants ---
    INFLATION_RATE = 0.07
    INITIAL_WITHDRAWAL = 100000
    LIFESPAN = 100
    CURRENT_YEAR = 2025

    # --- Prepare Data ---
    df = df_predictions.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['5th'] = df['5th'].replace({r'\$': '', ',': ''}, regex=True).astype(float)
    price_map = df.set_index('Year')['5th'].to_dict()

    # --- Data from the image's table ---
    image_data = {
        5: {2025: 10.96, 2030: 4.77, 2035: 2.77, 2040: 1.90, 2045: 1.44, 2050: 1.16, 2055: 0.97, 2060: 0.83, 2065: 0.72, 2070: 0.63, 2075: 0.56},
        15: {2025: 10.83, 2030: 4.64, 2035: 2.64, 2040: 1.77, 2045: 1.31, 2050: 1.03, 2055: 0.84, 2060: 0.70, 2065: 0.59, 2070: 0.50, 2075: 0.43},
        25: {2025: 10.71, 2030: 4.52, 2035: 2.52, 2040: 1.65, 2045: 1.19, 2050: 0.91, 2055: 0.72, 2060: 0.58, 2065: 0.47, 2070: 0.38, 2075: 0.31},
        35: {2025: 10.60, 2030: 4.41, 2035: 2.41, 2040: 1.54, 2045: 1.07, 2050: 0.79, 2055: 0.60, 2060: 0.47, 2065: 0.36, 2070: 0.27, 2075: 0.19},
        45: {2025: 10.47, 2030: 4.28, 2035: 2.28, 2040: 1.41, 2045: 0.95, 2050: 0.67, 2055: 0.48, 2060: 0.34, 2065: 0.23, 2070: 0.15, 2075: 0.07},
        55: {2025: 10.33, 2030: 4.14, 2035: 2.14, 2040: 1.27, 2045: 0.81, 2050: 0.53, 2055: 0.34, 2060: 0.20, 2065: 0.09, 2070: 0.00, 2075: 0.00},
        65: {2025: 10.13, 2030: 3.94, 2035: 1.94, 2040: 1.07, 2045: 0.61, 2050: 0.33, 2055: 0.14, 2060: 0.00, 2065: 0.00, 2070: 0.00, 2075: 0.00},
        75: {2025: 9.80, 2030: 3.61, 2035: 1.61, 2040: 0.74, 2045: 0.28, 2050: 0.00, 2055: 0.00, 2060: 0.00, 2065: 0.00, 2070: 0.00, 2075: 0.00}
    }

    html = "<h2>Verification of 'Bitcoin Needed to Retire'</h2>"
    html += "<p>This table verifies the numbers from the original analysis by calculating the total Bitcoin required to fund withdrawals from a specified retirement year until age 100, assuming a 7% annual inflation rate on a $100,000 initial withdrawal.</p>"

    for current_age, age_data in image_data.items():
        html += f"<h3>Verification for Current Age: {current_age}</h3>"
        html += """
        <style>
            table { width: 100%; border-collapse: collapse; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
        """
        html += "<table><tr><th>Retirement Year</th><th>Image BTC Needed</th><th>Calculated BTC Needed</th><th>Match</th></tr>"

        for retirement_year, btc_needed_image in age_data.items():
            retirement_age = current_age + (retirement_year - CURRENT_YEAR)
            years_in_retirement = LIFESPAN - retirement_age
            total_btc_needed = 0
            calculation_complete = True

            if years_in_retirement <= 0:
                total_btc_needed = 0.0
            else:
                for i in range(years_in_retirement):
                    year_of_withdrawal = retirement_year + i
                    if year_of_withdrawal not in price_map:
                        calculation_complete = False
                        break
                    inflation_adjusted_withdrawal = INITIAL_WITHDRAWAL * ((1 + INFLATION_RATE) ** i)
                    predicted_price = price_map[year_of_withdrawal]
                    total_btc_needed += inflation_adjusted_withdrawal / predicted_price

            if calculation_complete:
                match = "Yes" if abs(btc_needed_image - total_btc_needed) < 0.5 else "No"
                html += f"<tr><td>{retirement_year}</td><td>{btc_needed_image:.2f}</td><td>{total_btc_needed:.2f}</td><td>{match}</td></tr>"
            else:
                html += f"<tr><td>{retirement_year}</td><td>{btc_needed_image:.2f}</td><td>(Data ends before 100)</td><td>N/A</td></tr>"
        html += "</table>"
    return html

def create_interactive_bitcoin_chart(data, start_date):
    """Creates an interactive Bitcoin price chart with Plotly and appends a retirement table."""
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

    # --- Create CSV and Retirement Table ---
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

    retirement_html = generate_retirement_table_html(df_table)
    
    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    with open("interactive_bitcoin_chart.html", "w") as f:
        f.write("<html><head><title>Bitcoin Analysis</title></head><body>")
        f.write(chart_html)
        f.write(retirement_html)
        f.write("</body></html>")

    print("Interactive chart with retirement table saved to interactive_bitcoin_chart.html")

if __name__ == "__main__":
    start_date = "2010-07-17"
    end_date = datetime.today().strftime("%Y-%m-%d")
    
    price_data = get_bitcoin_data(start_date, end_date)
    if price_data:
        create_interactive_bitcoin_chart(price_data, start_date)
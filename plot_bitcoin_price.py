import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import statsmodels.api as sm

def get_bitcoin_data(start_date, end_date):
    """Fetches Bitcoin price data from the CryptoCompare API."""
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    
    all_data = []
    current_timestamp = end_timestamp

    while True:
        # We are fetching data backwards in time, so toTs will be updated in each iteration
        url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=2000&toTs={current_timestamp}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data['Response'] == 'Success' and 'Data' in data and 'Data' in data['Data']:
                new_data = data['Data']['Data']
                if not new_data:
                    break  # No more data available from the API

                # The API returns data in ascending order of time.
                # The first element is the oldest in the current chunk.
                oldest_timestamp_in_chunk = new_data[0]['time']

                # Add the new data to our list
                all_data.extend(new_data)

                # Set the timestamp for the next iteration to be the timestamp of the oldest record.
                # This will make the next request fetch data from before this point in time.
                current_timestamp = oldest_timestamp_in_chunk

                # If the oldest data point is already before our desired start date, we can stop.
                if oldest_timestamp_in_chunk <= start_timestamp:
                    break
            else:
                print(f"Error in API response: {data.get('Message', 'Unknown error')}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
            
    return all_data

def plot_bitcoin_price(data, start_date):
    """Plots the Bitcoin price data."""
    if not data:
        print("No data to plot.")
        return
        
    df = pd.DataFrame(data)
    # Convert timestamp to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Filter out data before the start date (as we might have fetched some extra)
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    df = df[df['time'] >= start_datetime]

    # Sort by date and remove duplicates
    df = df.sort_values('time').drop_duplicates(subset=['time'], keep='first')

    # Convert dates to number of days from start for log-log plot
    df['days_since_start'] = (df['time'] - start_datetime).dt.days + 1

    # Filter data to start from day 365
    df = df[df['days_since_start'] >= 365]

    # --- Quantile Regression ---
    X = sm.add_constant(np.log(df['days_since_start']))
    y = np.log(df['close'])

    # Fit the median regression (quantile=0.5)
    res_median = sm.QuantReg(y, X).fit(q=0.5)
    
    # Fit the upper and lower quantile regressions
    res_upper_95 = sm.QuantReg(y, X).fit(q=0.95)
    res_lower_05 = sm.QuantReg(y, X).fit(q=0.05)
    res_upper_99 = sm.QuantReg(y, X).fit(q=0.99)
    res_lower_10 = sm.QuantReg(y, X).fit(q=0.10)
    res_upper_90 = sm.QuantReg(y, X).fit(q=0.90)
    res_lower_30 = sm.QuantReg(y, X).fit(q=0.30)
    
    # --- Extend Lines ---
    future_days = (datetime(2120, 1, 1) - start_datetime).days + 1
    extended_days = np.arange(df['days_since_start'].min(), future_days)
    extended_X = sm.add_constant(np.log(extended_days))
    
    df_extended = pd.DataFrame({
        'days_since_start': extended_days,
        'fit': np.exp(res_median.predict(extended_X)),
        'upper_bound_95': np.exp(res_upper_95.predict(extended_X)),
        'lower_bound_05': np.exp(res_lower_05.predict(extended_X)),
        'upper_bound_99': np.exp(res_upper_99.predict(extended_X))
    })
    
    # --- R-squared ---
    r2 = r2_score(y, res_median.predict(X))
    # --- End R-squared ---

    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(df['days_since_start'], df['close'], label='Price')
    ax.plot(df_extended['days_since_start'], df_extended['fit'], label=f'Median Fit (R²={r2:.2f})', linestyle='--')
    ax.plot(df_extended['days_since_start'], df_extended['upper_bound_95'], label='95th Percentile', linestyle=':')
    ax.plot(df_extended['days_since_start'], df_extended['lower_bound_05'], label='5th Percentile', linestyle=':')
    ax.plot(df_extended['days_since_start'], df_extended['upper_bound_99'], label='99th Percentile', linestyle='-.')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # --- Custom Ticks for Halving Events ---
    halving_dates = [
        datetime(2012, 11, 28),
        datetime(2016, 7, 9),
        datetime(2020, 5, 11),
        datetime(2024, 4, 20)
    ]
    halving_days = [(d - start_datetime).days + 1 for d in halving_dates]

    # Generate a range of logarithmically spaced ticks
    log_ticks = [10**i for i in np.arange(np.log10(df['days_since_start'].min()), np.log10((datetime(2100, 1, 1) - start_datetime).days + 1), 0.1)]
    
    # Combine halving dates and log ticks
    all_ticks = sorted(list(set(log_ticks + halving_days)))
    
    # Filter ticks to only be within the plotted range
    all_ticks = [tick for tick in all_ticks if tick >= 365]

    ax.set_xticks(all_ticks)
    
    # Set x-axis limit
    ax.set_xlim(right=(datetime(2100, 1, 1) - start_datetime).days + 1)
    # --- End Custom Ticks ---

    # This function will format the x-tick labels
    def days_to_date_formatter(x, pos):
        return (start_datetime + timedelta(days=int(x))).strftime('%Y-%m-%d')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(days_to_date_formatter))
    
    fig.autofmt_xdate(rotation=45, ha='right')

    ax.set_title('Bitcoin Price History - Log-Log Scale with Quantile Regression')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True, which="both", ls="--")
    ax.legend()
    
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
    
    plt.tight_layout()
    plt.savefig('bitcoin_price_chart.png')
    plt.show()

if __name__ == "__main__":
    start_date = "2010-07-17"
    end_date = datetime.today().strftime("%Y-%m-%d")
    
    price_data = get_bitcoin_data(start_date, end_date)
    # Pass start_date to the plot function for filtering
    plot_bitcoin_price(price_data, start_date)

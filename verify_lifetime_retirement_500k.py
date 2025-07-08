import pandas as pd
from datetime import datetime

def verify_lifetime_retirement_500k():
    """
    Verifies the 'Bitcoin Needed to Retire' numbers from the 500k/year image
    using a lifetime withdrawal model with inflation.
    """
    # --- Constants ---
    INFLATION_RATE = 0.07
    INITIAL_WITHDRAWAL = 500000  # Updated for the 500k/year scenario
    LIFESPAN = 100
    CURRENT_YEAR = 2025 # Based on the image date 4/30/2025

    # --- Load and Prepare Data ---
    try:
        df = pd.read_csv('bitcoin_price_predictions.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['5th'] = df['5th'].replace({'\$': '', ',': ''}, regex=True).astype(float)
        price_map = df.set_index('Year')['5th'].to_dict()
    except FileNotFoundError:
        print("Error: 'bitcoin_price_predictions.csv' not found.")
        return

    # --- Data from the 500k/year image's table ---
    image_data = {
        5:  {2025: 54.81, 2030: 23.85, 2035: 13.85, 2040: 9.50, 2045: 7.20, 2050: 5.79, 2055: 4.85, 2060: 4.15, 2065: 3.61, 2070: 3.17, 2075: 2.78},
        15: {2025: 54.16, 2030: 23.20, 2035: 13.20, 2040: 8.85, 2045: 6.55, 2050: 5.15, 2055: 4.20, 2060: 3.51, 2065: 2.96, 2070: 2.52, 2075: 2.13},
        25: {2025: 53.57, 2030: 22.61, 2035: 12.61, 2040: 8.26, 2045: 5.95, 2050: 4.55, 2055: 3.61, 2060: 2.91, 2065: 2.37, 2070: 1.92, 2075: 1.54},
        35: {2025: 52.99, 2030: 22.03, 2035: 12.03, 2040: 7.68, 2045: 5.37, 2050: 3.97, 2055: 3.02, 2060: 2.33, 2065: 1.79, 2070: 1.34, 2075: 0.96},
        45: {2025: 52.37, 2030: 21.41, 2035: 11.41, 2040: 7.06, 2045: 4.76, 2050: 3.36, 2055: 2.41, 2060: 1.71, 2065: 1.17, 2070: 0.73, 2075: 0.34},
        55: {2025: 51.65, 2030: 20.68, 2035: 10.69, 2040: 6.34, 2045: 4.03, 2050: 2.63, 2055: 1.68, 2060: 0.99, 2065: 0.45, 2070: 0.00, 2075: 0.00},
        65: {2025: 50.66, 2030: 19.70, 2035: 9.70,  2040: 5.35, 2045: 3.04, 2050: 1.64, 2055: 0.69, 2060: 0.00, 2065: 0.00, 2070: 0.00, 2075: 0.00},
        75: {2025: 49.02, 2030: 18.06, 2035: 8.06,  2040: 3.71, 2045: 1.40, 2050: 0.00, 2055: 0.00, 2060: 0.00, 2065: 0.00, 2070: 0.00, 2075: 0.00}
    }

    for current_age, age_data in image_data.items():
        print(f"\n--- Verification for Current Age: {current_age} (500k/year) ---")
        print("-" * 80)
        print(f"{'Retirement Year':<18} {'Image BTC Needed':<18} {'Calculated BTC Needed':<25} {'Match':<10}")
        print("-" * 80)

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
                # Using a slightly larger tolerance due to higher numbers
                match = "Yes" if abs(btc_needed_image - total_btc_needed) < 0.25 else "No"
                print(f"{retirement_year:<18} {btc_needed_image:<18.2f} {total_btc_needed:<25.2f} {match:<10}")
            else:
                print(f"{retirement_year:<18} {btc_needed_image:<18.2f} {'(Data ends before 100)':<25} {'N/A':<10}")

    print("\nNote: 'Calculated BTC Needed' is the total BTC required to fund withdrawals")
    print("from retirement until age 100, with 7% annual inflation for a 500k/year lifestyle.")

if __name__ == "__main__":
    verify_lifetime_retirement_500k() 
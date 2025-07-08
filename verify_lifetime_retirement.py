import pandas as pd
from datetime import datetime

def verify_lifetime_retirement():
    """
    Verifies the 'Bitcoin Needed to Retire' numbers from the image
    using a lifetime withdrawal model with inflation.
    """
    # --- Constants ---
    INFLATION_RATE = 0.07
    INITIAL_WITHDRAWAL = 100000
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

    for current_age, age_data in image_data.items():
        print(f"\n--- Verification for Current Age: {current_age} ---")
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
                match = "Yes" if abs(btc_needed_image - total_btc_needed) < 0.5 else "No"
                print(f"{retirement_year:<18} {btc_needed_image:<18.2f} {total_btc_needed:<25.2f} {match:<10}")
            else:
                print(f"{retirement_year:<18} {btc_needed_image:<18.2f} {'(Data ends before 100)':<25} {'N/A':<10}")

    print("\nNote: 'Calculated BTC Needed' is the total BTC required to fund withdrawals")
    print("from retirement until age 100, with 7% annual inflation.")

if __name__ == "__main__":
    verify_lifetime_retirement()

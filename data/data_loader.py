import requests
import pandas as pd
from datetime import datetime
import config

class DataLoader:
    def __init__(self):
        self.api_key = config.ALPHA_VANTAGE_API_KEY
        self.base_url = config.BASE_URL

    def get_stock_data(self, symbol, interval='daily', output_size='full', market='USD'):
        """
        Retrieve stock data from the Alpha Vantage API.
        """
        params = {
            'function': f'TIME_SERIES_{interval.upper()}',
            'symbol': symbol,
            'market': market,
            'apikey': self.api_key,
            'outputsize': output_size
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        # Check for API response errors
        if "Error Message" in data:
            raise ValueError(f"API Error: {data['Error Message']}")

        # Determine the correct time series key based on the interval
        time_series_key = f'Time Series ({interval.capitalize()})'
        
        if time_series_key not in data:
            raise KeyError(f"Expected key '{time_series_key}' not found in response. Please check the stock symbol and API settings.")
        
        # Process the data into a DataFrame
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        
        # Rename columns for clarity
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Convert column data types to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        return df

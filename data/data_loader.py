import requests
import config
import pandas as pd

class DataLoader:
    def __init__(self):
        self.api_key = config.TWELVE_DATA_API_KEY
        self.base_url = "https://api.twelvedata.com"

    def get_stock_data(self, symbol, interval='1day', outputsize='5000', output_format='pandas'):
        """
        Retrieve stock data from the Twelve Data API.
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize,
            'apikey': self.api_key,
            'output': output_format
        }

        response = requests.get(f"{self.base_url}/time_series", params=params)
        data = response.json()
        print(data)  # Kiểm tra cấu trúc dữ liệu API trả về

        # Check for API response errors
        if "message" in data:
            raise ValueError(f"API Error: {data['message']}")

        # Kiểm tra và xử lý dữ liệu
        if 'values' not in data or not data['values']:
            raise ValueError(f"No stock data found for symbol: {symbol}")

        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        
        # Chuyển đổi các cột sang kiểu số
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Loại bỏ các hàng có giá trị NaN
        df = df.dropna(subset=numeric_columns)
        
        if df.empty:
            raise ValueError(f"No valid stock data found for symbol: {symbol}")
        
        return df
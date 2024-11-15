import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def create_features(self, data):
        """Create technical indicators for prediction"""
        df = data.copy()
        
        # Basic technical indicators
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['close'])
        df['MACD'] = self.calculate_macd(df['close'])
        df['VOL_MA'] = df['volume'].rolling(window=10).mean()
        
        # Additional technical indicators
        df['Bollinger_Upper'], df['Bollinger_Lower'] = self.calculate_bollinger_bands(df['close'])
        df['ATR'] = self.calculate_atr(df['high'], df['low'], df['close'])
        df['OBV'] = self.calculate_obv(df['close'], df['volume'])
        
        # Price changes and momentum
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(periods=5)
        df['price_change_20'] = df['close'].pct_change(periods=20)
        df['momentum'] = df['close'] / df['close'].shift(5) - 1
        
        # Volatility features
        df['volatility'] = df['price_change'].rolling(window=20).std()
        df['volume_change'] = df['volume'].pct_change()
        
        # Target variable
        df['future_change'] = df['price_change'].shift(-1)
        
        return df.dropna()
    
    def prepare_data(self, data, prediction_days):
        """Prepare features and target for training"""
        df = data.copy()
        
        # Features for prediction
        feature_columns = [
            'SMA_5', 'SMA_20', 'RSI', 'MACD', 'VOL_MA',
            'Bollinger_Upper', 'Bollinger_Lower', 'ATR', 'OBV',
            'price_change', 'price_change_5', 'price_change_20',
            'momentum', 'volatility', 'volume_change'
        ]
        
        X = df[feature_columns]
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
        
        # Target: 1 if price goes up, 0 if price goes down
        y = (df['future_change'] > 0).astype(int)
        
        return X_scaled, y
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band
    
    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def calculate_obv(self, close, volume):
        """Calculate On-Balance Volume"""
        price_change = close.diff()
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        return obv
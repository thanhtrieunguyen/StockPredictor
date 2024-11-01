import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def create_features(self, data):
        """Create technical indicators for prediction"""
        df = data.copy()
        
        # Technical indicators
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['close'])
        df['MACD'] = self.calculate_macd(df['close'])
        df['VOL_MA'] = df['volume'].rolling(window=10).mean()
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['future_change'] = df['price_change'].shift(-1)
        
        return df.dropna()
    
    def prepare_data(self, data, prediction_days):
        """Prepare features and target for training"""
        df = data.copy()
        
        # Features for prediction
        feature_columns = ['SMA_5', 'SMA_20', 'RSI', 'MACD', 'VOL_MA', 'price_change']
        X = df[feature_columns]
        
        # Target: 1 if price goes up, 0 if price goes down
        y = (df['future_change'] > 0).astype(int)
        
        return X, y
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        return exp1 - exp2
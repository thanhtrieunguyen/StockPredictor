import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

class StockDataManager:
    def __init__(self, data_dir="stock_data"):
        """
        Khởi tạo Stock Data Manager
        
        Parameters:
        data_dir (str): Thư mục lưu trữ dữ liệu chứng khoán
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def get_stock_file_path(self, symbol):
        """Lấy đường dẫn file CSV cho một mã chứng khoán"""
        return self.data_dir / f"{symbol.upper()}.csv"
    
    def load_existing_data(self, symbol):
        """
        Tải dữ liệu hiện có của một mã chứng khoán
        
        Returns:
        pd.DataFrame hoặc None nếu file không tồn tại
        """
        file_path = self.get_stock_file_path(symbol)
        if file_path.exists():
            df = pd.read_csv(file_path, index_col='date', parse_dates=['date'])
            return df.sort_index()
        return None
    
    def get_required_date_range(self, existing_data, months_back=6):
        """
        Xác định khoảng thời gian cần lấy dữ liệu
        
        Returns:
        tuple: (start_date, end_date)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * months_back)
        
        return start_date, end_date
    
    def update_stock_data(self, symbol, data_loader, months_back=6):
        """
        Cập nhật dữ liệu chứng khoán
        
        Parameters:
        symbol (str): Mã chứng khoán
        data_loader (DataLoader): Instance của DataLoader để lấy dữ liệu
        months_back (int): Số tháng dữ liệu lịch sử cần lấy
        
        Returns:
        pd.DataFrame: Dữ liệu đã cập nhật
        """
        # Lấy dữ liệu hoàn toàn mới
        new_data = data_loader.get_stock_data(symbol)
        
        # Lọc dữ liệu trong khoảng thời gian cần thiết
        start_date, end_date = self.get_required_date_range(None, months_back)
        new_data = new_data[(new_data.index >= start_date) & (new_data.index <= end_date)]
        
        # Sắp xếp theo ngày
        new_data = new_data.sort_index()
        
        # Lưu dữ liệu
        self.save_stock_data(symbol, new_data)
        
        return new_data
    
    def save_stock_data(self, symbol, data):
        """Lưu dữ liệu vào file CSV"""
        file_path = self.get_stock_file_path(symbol)
        data.to_csv(file_path, index=True, index_label='date')
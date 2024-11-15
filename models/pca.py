from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

class PCAModel:
    def __init__(self, n_components=None, variance_threshold=0.95):
        """
        Khởi tạo PCA Model
        
        Parameters:
        n_components: int, số lượng components muốn giữ lại
        variance_threshold: float, ngưỡng phương sai tích lũy (nếu n_components=None)
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca = None
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train=None):
        """
        Huấn luyện mô hình PCA
        
        Parameters:
        X_train: ma trận features
        y_train: không sử dụng trong PCA nhưng giữ để thống nhất interface
        """
        # Chuẩn hóa dữ liệu
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Nếu n_components chưa được chỉ định, tự động xác định dựa vào variance_threshold
        if self.n_components is None:
            self.pca = PCA()
            self.pca.fit(X_scaled)
            
            # Tính toán số lượng components cần thiết
            cumsum = np.cumsum(self.pca.explained_variance_ratio_)
            self.n_components = np.argmax(cumsum >= self.variance_threshold) + 1
            
            # Khởi tạo lại PCA với số components phù hợp
            self.pca = PCA(n_components=self.n_components)
            
        else:
            self.pca = PCA(n_components=self.n_components)
            
        self.pca.fit(X_scaled)
        return self
    
    def transform(self, X):
        """
        Áp dụng PCA transformation lên dữ liệu mới
        """
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def inverse_transform(self, X_transformed):
        """
        Chuyển đổi ngược từ không gian PCA về không gian gốc
        """
        X_original_scaled = self.pca.inverse_transform(X_transformed)
        return self.scaler.inverse_transform(X_original_scaled)
    
    def predict(self, X):
        """
        Interface dự đoán để tương thích với scikit-learn
        Trong trường hợp này, nó chỉ đơn giản transform dữ liệu
        """
        return self.transform(X)
    
    def get_feature_importance(self):
        """
        Trả về tầm quan trọng của các features dựa trên explained variance ratio
        """
        return {
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(self.pca.explained_variance_ratio_),
            'n_components': self.n_components,
            'components': self.pca.components_
        }
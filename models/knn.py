from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

class KNNModel:
    def __init__(self):
        """
        Khởi tạo KNN Model với scaling
        """
        self.model = None
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train):
        """
        Huấn luyện mô hình KNN với Grid Search để tìm tham số tối ưu
        """
        # Chuẩn hóa dữ liệu
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Định nghĩa grid search parameters
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree']
        }
        
        # Khởi tạo mô hình và thực hiện grid search
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(
            knn,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Huấn luyện mô hình
        grid_search.fit(X_scaled, y_train)
        self.model = grid_search.best_estimator_
        
        return self
    
    def predict(self, X):
        """
        Dự đoán cho dữ liệu mới
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Dự đoán xác suất cho dữ liệu mới
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_neighbors(self, X, n_neighbors=None):
        """
        Lấy các điểm dữ liệu gần nhất cho một mẫu
        
        Parameters:
        X: dữ liệu cần tìm neighbors
        n_neighbors: số lượng neighbors muốn lấy (mặc định là K của mô hình)
        """
        X_scaled = self.scaler.transform(X)
        if n_neighbors is None:
            n_neighbors = self.model.n_neighbors
        
        distances, indices = self.model.kneighbors(
            X_scaled,
            n_neighbors=n_neighbors,
            return_distance=True
        )
        
        return distances, indices
    
    def get_best_params(self):
        """
        Trả về các tham số tốt nhất của mô hình
        """
        return self.model.get_params()
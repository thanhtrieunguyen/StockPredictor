from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np

class LogisticRegressionModel:
    def __init__(self):
        """
        Khởi tạo Logistic Regression Model với scaling
        """
        self.model = None
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train):
        """
        Huấn luyện mô hình Logistic Regression với Grid Search
        """
        # Chuẩn hóa dữ liệu
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Định nghĩa grid search parameters
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        }
        
        # Khởi tạo model và thực hiện grid search
        lr = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(
            lr,
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
    
    def get_feature_importance(self, feature_names):
        """
        Tính toán và trả về độ quan trọng của các features
        """
        coefficients = self.model.coef_[0]
        
        # Lấy giá trị tuyệt đối của các hệ số
        importances = np.abs(coefficients)
        
        # Chuẩn hóa tầm quan trọng
        normalized_importances = importances / np.sum(importances)
        
        # Tạo dictionary ánh xạ feature với importance
        feature_importance = dict(zip(feature_names, normalized_importances))
        
        return feature_importance
    
    def get_model_params(self):
        """
        Trả về các tham số của mô hình
        """
        return {
            'best_params': self.model.get_params(),
            'coef': self.model.coef_.tolist(),
            'intercept': self.model.intercept_.tolist(),
            'classes': self.model.classes_.tolist(),
            'n_features': self.model.n_features_in_
        }
    
    def get_decision_boundary(self, X):
        """
        Tính toán decision boundary cho visualization
        """
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
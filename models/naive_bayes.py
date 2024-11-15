from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import numpy as np

class NaiveBayesModel:
    def __init__(self):
        """
        Khởi tạo Naive Bayes Model với scaling
        """
        self.model = GaussianNB()
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train):
        """
        Huấn luyện mô hình Naive Bayes
        """
        # Chuẩn hóa dữ liệu
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Huấn luyện mô hình
        self.model.fit(X_scaled, y_train)
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
    
    def get_model_params(self):
        """
        Trả về các tham số của mô hình
        """
        return {
            'class_prior': self.model.class_prior_,
            'classes': self.model.classes_,
            'n_features': self.model.n_features_in_,
            'feature_means': self.model.theta_,
            'feature_variances': self.model.var_
        }
    
    def get_feature_importance(self, feature_names):
        """
        Tính toán độ quan trọng của features dựa trên variance
        """
        importances = np.mean(self.model.var_, axis=0)
        total_importance = np.sum(importances)
        
        # Chuẩn hóa tầm quan trọng
        normalized_importances = importances / total_importance
        
        # Tạo dictionary ánh xạ feature với importance
        feature_importance = dict(zip(feature_names, normalized_importances))
        
        return feature_importance
    
    def evaluate_priors(self):
        """
        Đánh giá các prior probabilities của các classes
        """
        return dict(zip(self.model.classes_, self.model.class_prior_))
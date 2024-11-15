from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

class RandomForestModel:
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def train(self, X_train, y_train):
        """
        Huấn luyện model với Grid Search CV
        """
        self.feature_names = X_train.columns.tolist()
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'criterion': ['gini', 'entropy']
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        return self.model
    
    def predict(self, X):
        """
        Dự đoán nhãn cho dữ liệu mới
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Dự đoán xác suất cho dữ liệu mới
        """
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names=None):
        """
        Lấy độ quan trọng của các features
        """
        if feature_names is None:
            feature_names = self.feature_names
            
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'std': std
        })
        return feature_imp.sort_values('importance', ascending=False)
    
    def get_model_params(self):
        """
        Lấy các tham số của model
        """
        return {
            'best_params': self.model.get_params(),
            'n_features': self.model.n_features_in_,
            'n_estimators': self.model.n_estimators,
            'oob_score': self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
        }
    
    def get_estimator_predictions(self, X):
        """
        Lấy dự đoán từ từng cây trong rừng
        """
        predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        return predictions
    
    def get_feature_importance_by_class(self):
        """
        Tính toán feature importance cho từng class
        """
        importances_by_class = {}
        for i, class_label in enumerate(self.model.classes_):
            class_importances = []
            for tree in self.model.estimators_:
                class_importances.append(
                    tree.feature_importances_ * (tree.predict_proba(X)[:, i].mean())
                )
            importances_by_class[class_label] = np.mean(class_importances, axis=0)
        
        return importances_by_class
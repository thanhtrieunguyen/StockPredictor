from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DecisionTreeModel:
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def train(self, X_train, y_train):
        """
        Huấn luyện model với Grid Search CV
        """
        self.feature_names = X_train.columns.tolist()
        
        param_grid = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random']
        }
        
        dt = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
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
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        return feature_imp.sort_values('importance', ascending=False)
    
    def plot_tree_structure(self, figsize=(20,10)):
        """
        Vẽ cấu trúc của decision tree
        """
        plt.figure(figsize=figsize)
        plot_tree(self.model, 
                 feature_names=self.feature_names,
                 class_names=['Down', 'Up'],
                 filled=True,
                 rounded=True)
        plt.show()
    
    def get_decision_path(self, X):
        """
        Lấy đường đi quyết định cho một mẫu dữ liệu
        """
        paths = self.model.decision_path(X)
        return paths
    
    def get_model_params(self):
        """
        Lấy các tham số của model
        """
        return {
            'best_params': self.model.get_params(),
            'n_features': self.model.n_features_in_,
            'tree_depth': self.model.get_depth(),
            'n_leaves': self.model.get_n_leaves()
        }
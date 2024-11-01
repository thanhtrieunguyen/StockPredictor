from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

class RandomForestModel:
    def __init__(self):
        self.model = None
        
    def train(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance scores"""
        importances = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        return feature_imp.sort_values('importance', ascending=False)
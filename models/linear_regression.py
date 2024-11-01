from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

class LogisticRegressionModel:
    def __init__(self):
        self.model = None
        
    def train(self, X_train, y_train):
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        }
        
        lr = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
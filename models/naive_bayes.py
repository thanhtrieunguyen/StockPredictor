from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

class NaiveBayesModel:
    def __init__(self):
        self.model = None
        
    def train(self, X_train, y_train):
        param_grid = {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
        
        nb = GaussianNB()
        grid_search = GridSearchCV(nb, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
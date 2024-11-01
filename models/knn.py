from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

class KNNModel:
    def __init__(self):
        self.model = None
        
    def train(self, X_train, y_train):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
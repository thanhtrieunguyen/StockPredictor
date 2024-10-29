from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

class DecisionTreeModel:
    def __init__(self):
        self.model = None
        
    def train(self, X_train, y_train):
        param_grid = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        dt = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
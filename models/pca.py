from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

class PCAModel:
    def __init__(self):
        self.model = None
        
    def train(self, X_train, y_train):
        # Create a pipeline with PCA and Random Forest
        pipeline = Pipeline([
            ('pca', PCA()),
            ('rf', RandomForestClassifier(random_state=42))
        ])
        
        param_grid = {
            'pca__n_components': [2, 3, 4],
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [3, 5, 7],
            'rf__min_samples_split': [2, 5],
            'rf__min_samples_leaf': [1, 2]
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
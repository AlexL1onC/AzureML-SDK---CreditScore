# models/model.py
from xgboost import XGBClassifier

class XGBoostModel:
    def __init__(self, params=None):
        if params is None:
            params = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "objective": "binary:logistic",
                "eval_metric": "logloss"
            }
        
        self.model = XGBClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
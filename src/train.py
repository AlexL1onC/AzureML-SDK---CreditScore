import os
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

from src.database import SQLDataHandler

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "../artifacts")
FEATURE_COLS = [
    "UnitPriceDiscount",
    "LineTotal",
    "ProductCategoryID",
    "UnitPrice",
    "day_of_week",
    "month",
    "year",
    "day_of_month",
    "is_weekend",
    "ship_method_encoded",
]
TARGET = "OrderVolume"


class OrderClassifier:
    def __init__(self, config_path, artifacts_dir=ARTIFACTS_DIR, random_state=42):
        self.config_path = config_path
        self.artifacts_dir = artifacts_dir
        self.random_state = random_state
        self.db_handler = SQLDataHandler(config_file=config_path)
        self.model = None
        self.best_params = None
        self.metrics = {}

    def load_data(self):
        query = """
            SELECT
                d.OrderQty,
                d.UnitPrice,
                d.UnitPriceDiscount,
                d.LineTotal,
                p.ProductCategoryID,
                h.OrderDate,
                h.OnlineOrderFlag,
                h.ShipMethod
            FROM SalesLT.SalesOrderDetail d
            JOIN SalesLT.SalesOrderHeader h  ON d.SalesOrderID  = h.SalesOrderID
            JOIN SalesLT.Product p           ON d.ProductID     = p.ProductID
        """
        df = self.db_handler.fetch_data(query)
        df["OrderDate"] = pd.to_datetime(df["OrderDate"])
        return df

    def preprocess(self, df):
        df = df.copy()

        df["OrderVolume"] = pd.qcut(df["OrderQty"], q=4, labels=[0, 1, 2, 3])
        df["OrderVolume"] = df["OrderVolume"].astype(int)

        df["day_of_week"] = df["OrderDate"].dt.dayofweek
        df["month"] = df["OrderDate"].dt.month
        df["year"] = df["OrderDate"].dt.year
        df["day_of_month"] = df["OrderDate"].dt.day
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        df["ship_method_encoded"] = df["ShipMethod"].astype("category").cat.codes
        df["ProductCategoryID"] = df["ProductCategoryID"].fillna(-1).astype(int)

        return df

    def train(self, df):
        X = df[FEATURE_COLS]
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y,
        )

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        estimator = XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )

        grid = GridSearchCV(
            estimator,
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=2,
        )
        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

        y_pred = self.model.predict(X_test)
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }

        print(f"\nBest params: {self.best_params}")
        print(f"Test accuracy: {self.metrics['accuracy']:.4f}")

        return self.model

    def save_artifacts(self):
        os.makedirs(self.artifacts_dir, exist_ok=True)

        model_path = os.path.join(self.artifacts_dir, "order_classifier.pkl")
        joblib.dump(self.model, model_path)
        print(f"Model saved        -> {model_path}")

        metrics_path = os.path.join(self.artifacts_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": self.metrics, "best_params": self.best_params}, f, indent=4)
        print(f"Metrics saved      -> {metrics_path}")

        fi_df = pd.DataFrame(
            {"feature": FEATURE_COLS, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)
        fi_path = os.path.join(self.artifacts_dir, "feature_importance.csv")
        fi_df.to_csv(fi_path, index=False)
        print(f"Feature importance -> {fi_path}")

    def close_connection(self):
        self.db_handler.close_connection()


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "../API/config.json")
    classifier = OrderClassifier(config_path=config_path)

    print("Loading data...")
    df = classifier.load_data()
    print(f"Rows: {len(df)}  |  Unique dates: {df['OrderDate'].nunique()}")
    print(f"OrderQty stats:\n{df['OrderQty'].describe()}\n")

    print("Preprocessing data...")
    df = classifier.preprocess(df)

    print("Training classifier...")
    classifier.train(df)

    print("Saving artifacts...")
    classifier.save_artifacts()

    classifier.close_connection()
    print("Training completed.")

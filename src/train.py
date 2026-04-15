from database import SQLDataHandler
import os
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "../artifacts")


def load_data(db):
    """
    Each row = one order line item.
    Joins SalesOrderDetail -> SalesOrderHeader -> ProductCategory for features.
    """
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
    df = db.fetch_data(query)
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    return df


def engineer_features(df):
    df = df.copy()

    # Calendar features from OrderDate
    df["day_of_week"]  = df["OrderDate"].dt.dayofweek
    df["month"]        = df["OrderDate"].dt.month
    df["year"]         = df["OrderDate"].dt.year
    df["day_of_month"] = df["OrderDate"].dt.day
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    # Encode ShipMethod
    df["ship_method_encoded"] = df["ShipMethod"].astype("category").cat.codes

    # Fill any nulls in ProductCategoryID
    df["ProductCategoryID"] = df["ProductCategoryID"].fillna(-1).astype(int)

    return df


FEATURE_COLS = [
    "UnitPriceDiscount", "LineTotal", "ProductCategoryID", "UnitPrice",
]
TARGET = "OrderQty"


def run_grid_search(X_train, y_train):
    param_grid = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 5, 7],
        "learning_rate":    [0.05, 0.1, 0.2],
        "subsample":        [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb  = XGBRegressor(objective="reg:squarederror", random_state=42)
    grid = GridSearchCV(
        xgb,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X_train, y_train)
    print(f"\nBest params: {grid.best_params_}")
    return grid.best_estimator_, grid.best_params_


def save_artifacts(model, metrics, best_params):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved        -> {model_path}")

    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"metrics": metrics, "best_params": best_params}, f, indent=4)
    print(f"Metrics saved      -> {metrics_path}")

    fi_df = (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"Feature importance -> {fi_path}")


if __name__ == "__main__":
    # 1. Connect
    config_path = os.path.join(os.path.dirname(__file__), "../API/config.json")
    db = SQLDataHandler(config_file=config_path)

    # 2. Load
    print("Loading data...")
    df = load_data(db)
    print(f"Rows: {len(df)}  |  Unique dates: {df['OrderDate'].nunique()}")
    print(f"Target stats:\n{df[TARGET].describe()}\n")

    # 3. Feature engineering
    df = engineer_features(df)

    X = df[FEATURE_COLS]
    y = df[TARGET]

    # 4. Train/test split (stratified by OrderQty not needed; simple split fine here)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")

    # 5. Grid search
    print("\nRunning grid search...")
    best_model, best_params = run_grid_search(X_train, y_train)

    # 6. Evaluate
    y_pred = best_model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae":  float(mean_absolute_error(y_test, y_pred)),
        "r2":   float(r2_score(y_test, y_pred)),
    }
    print(f"\nTest metrics:")
    print(f"  RMSE : {metrics['rmse']:.4f}")
    print(f"  MAE  : {metrics['mae']:.4f}")
    print(f"  R2   : {metrics['r2']:.4f}")

    # 7. Save artifacts
    save_artifacts(best_model, metrics, best_params)

    db.close_connection()

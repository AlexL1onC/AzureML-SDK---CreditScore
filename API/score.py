import json
import joblib
import pandas as pd
import os
import numpy as np
from azureml.core.model import Model

def init():
    global model
    # El nombre debe coincidir con el que registres en deployment.py
    model_path = Model.get_model_path('order_classifier') 
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        df = pd.DataFrame(data)
        
        # 1. Replicar Ingeniería de Variables de train.py
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
        df["day_of_week"]  = df["OrderDate"].dt.dayofweek
        df["month"]        = df["OrderDate"].dt.month
        df["year"]         = df["OrderDate"].dt.year
        df["day_of_month"] = df["OrderDate"].dt.day
        df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)
        
        # Mapeo manual de ShipMethod (debe ser igual al de entrenamiento)
        # Si usaste LabelEncoder, lo ideal sería cargarlo aquí también
        ship_method_map = {'Cargo Transport 1': 0, 'Express Logistics': 1, 'Standard Post': 2}
        df['ship_method_encoded'] = df['ShipMethod'].map(ship_method_map).fillna(-1)

        # 2. Seleccionar solo las columnas que el modelo conoce
        FEATURE_COLS = [
            "UnitPriceDiscount", "LineTotal", "ProductCategoryID", "UnitPrice",
            "day_of_week", "month", "year", "day_of_month", "is_weekend", "ship_method_encoded"
        ]
        
        # 3. Predicción de Cuartil (0, 1, 2, 3)
        predictions = model.predict(df[FEATURE_COLS])
        
        # Mapeo de labels para que la respuesta sea cualitativa
        labels = {0: "Volumen Muy Bajo", 1: "Volumen Bajo", 2: "Volumen Medio", 3: "Volumen Alto"}
        result = [labels.get(p, "Desconocido") for p in predictions]
        
        return json.dumps({
            "predictions": predictions.tolist(),
            "labels": result
        })
    except Exception as e:
        return json.dumps({"error": str(e)})
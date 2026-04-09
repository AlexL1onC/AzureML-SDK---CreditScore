import json
import joblib
import pandas as pd
import os
from azureml.core.model import Model

def init():
    global model
    # Buscamos el modelo registrado en Azure
    model_path = Model.get_model_path('modelo_vervena') 
    model = joblib.load(model_path)

def run(raw_data):
    try:
        # 1. Recibir datos
        data = json.loads(raw_data)['data'][0]
        df = pd.DataFrame(data)
        
        # 2. Limpieza (Igual a la del entrenamiento)
        # Asegúrate que estas columnas coincidan con lo que discutimos
        cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
        df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        # 3. Predicción
        # Aquí podrías necesitar aplicar dummies si no se hizo antes
        prediction = model.predict(df_clean)
        
        return json.dumps(prediction.tolist())
    except Exception as e:
        return json.dumps({"error": str(e)})
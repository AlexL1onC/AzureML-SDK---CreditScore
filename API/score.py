import json
import joblib
import pandas as pd
import os
from azureml.core.model import Model
from src.zorrouno import processor

def init():
    global model
    global threshold
    model_path = Model.get_model_path('modelo_vervena') 
    model = joblib.load(model_path)
    
    # Leer umbral si existe el archivo, de lo contrario usar el estipulado
    threshold_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'umbral.json')
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            threshold = float(json.load(f).get('threshold', 0.7285))
    else:
        threshold = 0.7285

def run(raw_data):
    try:
        # Adaptado para poder recibir múltiples registros
        data = json.loads(raw_data)['data']
        df = pd.DataFrame(data)
        
        # 1. Transformación usando la clase solicitada
        df_clean = processor.embbed(df)
        
        # 2. Extracción de la probabilidad (no se aplica sigmoide extra a esto)
        # predict_proba retorna probabilidades [P(clase=0), P(clase=1)]
        probabilities = model.predict_proba(df_clean)[:, 1]
        
        # 3. Clasificación basada en el umbral
        predictions = (probabilities >= threshold).astype(int)
        
        return json.dumps({
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        })
    except Exception as e:
        return json.dumps({"error": str(e)})
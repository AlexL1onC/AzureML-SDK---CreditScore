from database import SQLDataHandler
from zorrouno import processor
from model import XGBoostModel
import joblib
import json

def run_training_pipeline():
    # 1. Obtención de datos desde Azure SQL
    # Tu clase SQLDataHandler ya maneja la conexión
    handler = SQLDataHandler(config_file="API/config.json") 
    query = "SELECT * FROM NombreDeTuTabla" 
    df_raw = handler.fetch_data(query)
    
    # 2. Limpieza y Selección de características
    # Usas el método estático de la clase que creaste
    df_clean = processor.embbed(df_raw)
    
    # 3. Preparación de variables
    # La variable objetivo es 'Exited' según la referencia
    X = df_clean.drop(columns=['Exited'])
    y = df_clean['Exited']
    
    # 4. Entrenamiento del modelo
    # Utilizas tu implementación de XGBoost
    model_wrapper = XGBoostModel()
    model_wrapper.fit(X, y)
    
    # 5. Guardado de artefactos
    # El archivo 'model.pkl' es el que registrarás en AzureML
    joblib.dump(model_wrapper.model, 'model.pkl')
    
    # Ejemplo de cálculo y guardado de umbral (threshold)
    # Es vital para tu script de score.py
    umbral_data = {"threshold": 0.7285} 
    with open('umbral.json', 'w') as f:
        json.dump(umbral_data, f)

    print("Entrenamiento completado y artefactos guardados.")

if __name__ == "__main__":
    run_training_pipeline()
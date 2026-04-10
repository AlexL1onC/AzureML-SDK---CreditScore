from src.database import SQLDataHandler
from src.model import ModelManager
from src.deployment import AzureDeployer
import json

if __name__ == "__main__":
    # 1. Cargar configuración (Evita hardcoding)
    with open('config.json') as f:
        config = json.load(f)

    # 2. Obtener Datos
    db = SQLDataHandler(config['server'], config['db'], config['user'], config['pwd'])
    datos = db.get_table_data("SalesLT.Customer") # Cambia por tu tabla

    # 3. Entrenar
    manager = ModelManager()
    acc = manager.train(datos, target_col="Exited") # Cambia por tu variable cualitativa
    manager.save_model("model.pkl")
    print(f"Modelo entrenado con precisión de: {acc}")

    # 4. Desplegar a la Nube
    deployer = AzureDeployer(config['sub_id'], config['rg'], config['ws_name'])
    uri = deployer.register_and_deploy("model.pkl", "modelo_vervena")
    
    print(f"--- DESPLIEGUE EXITOSO ---")
    print(f"Tu API está viva en: {uri}")
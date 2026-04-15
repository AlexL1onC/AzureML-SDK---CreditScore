from src.deployment import AzureDeployer
import json
import os

if __name__ == "__main__":
    # 1. Cargar configuración
    with open('API/config.json') as f:
        config = json.load(f)

    # 2. REFERENCIA AL MODELO PRE-ENTRENADO
    # Asegúrate de que este archivo ya exista en tu carpeta artifacts
    model_local_path = "artifacts/model.pkl"
    
    if os.path.exists(model_local_path):
        print(f"--- Iniciando despliegue del modelo pre-entrenado: {model_local_path} ---")
        
        # 3. Desplegar a la Nube
        # El AzureDeployer registrará el archivo .pkl local en el Workspace de Azure
        deployer = AzureDeployer(
            subscription_id=config['sub_id'], 
            resource_group=config['rg'], 
            workspace_name=config['ws_name']
        )
        
        # 'modelo_vervena' es el nombre que score.py buscará en la nube
        uri = deployer.register_and_deploy(model_local_path, "modelo_vervena")
        
        print(f"--- DESPLIEGUE EXITOSO ---")
        print(f"Tu API está viva en: {uri}")
    else:
        print("Error: No se encontró el archivo model.pkl. Ejecuta primero src/train.py localmente.")
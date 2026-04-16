from src.train import OrderClassifier
from src.deployment import AzureDeployer
import os

if __name__ == "__main__":
    # 1. Definir rutas
    root = os.path.dirname(os.path.abspath(__file__))
    config = os.path.join(root, "API", "config.json")
    
    # 2. Proceso de Entrenamiento
    #classifier = OrderClassifier(config_path=config)
    #df = classifier.load_data()
    
    #df = classifier.preprocess(df)

    # Entrenar y guardar (esto genera el .pkl y los JSON de métricas)
    #classifier.train(df)
    #accuracy = classifier.metrics["accuracy"]
    #classifier.save_artifacts()
    #print(f"Entrenamiento completado. Accuracy: {accuracy}")
    accuracy = 0.65 # Solo para pruebas, reemplaza con el valor real de tu entrenamiento

    # 3. Despliegue (Solo si el accuracy es aceptable)
    if accuracy > 0.6:
        deployer = AzureDeployer(config_path=config)
        # Asegúrate de que apunte al nuevo pkl
        uri = deployer.register_and_deploy(
        model_path="artifacts/order_classifier.pkl", 
        model_name="order_classifier"
        )
        print(f"API desplegada en: {uri}")
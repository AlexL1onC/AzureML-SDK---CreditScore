from azureml.core import Workspace, Environment, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.conda_dependencies import CondaDependencies
import json

class AzureDeployer:
    def __init__(self, config_path):
            # 1. Leer el archivo JSON
            with open(config_path, 'r') as f:
                c = json.load(f)
            
            # 2. Usar los valores del JSON para conectar a Azure
            # Asegúrate de que los nombres coincidan con tu JSON (sub_id, rg, ws_name)
            self.ws = Workspace.get(
                name=c['workspace_name'], 
                subscription_id=c['subscription_id'], 
                resource_group=c['resource_group']
            )

    def register_and_deploy(self, model_path, model_name="order_classifier"):
        # Registro del nuevo pkl
        model = Model.register(model_path=model_path, model_name=model_name, workspace=self.ws)
        
        env = Environment("order-env")
        env.python.conda_dependencies = CondaDependencies.create(
            conda_packages=['pandas', 'scikit-learn', 'numpy'],
            pip_packages=['xgboost', 'azureml-defaults']
        )
        
        inf_config = InferenceConfig(
            entry_script="API/score.py", 
            source_directory=".", # Esto sube todo el repo para que score.py vea las carpetas
            environment=env
        )
        
        aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
        
        service = Model.deploy(workspace=self.ws, name="order-service-final-v01", 
                               models=[model], inference_config=inf_config, 
                               deployment_config=aci_config, overwrite=True)
        
        service.wait_for_deployment(show_output=True)
        return service.scoring_uri
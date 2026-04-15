from azureml.core import Workspace, Environment, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.conda_dependencies import CondaDependencies
from src.zorrouno import processor

class AzureDeployer:
    def __init__(self, subscription_id, resource_group, workspace_name):
        self.ws = Workspace.get(name=workspace_name, 
                                subscription_id=subscription_id, 
                                resource_group=resource_group)

    def register_and_deploy(self, model_path, model_name):
        model = Model.register(model_path=model_path, model_name=model_name, workspace=self.ws)
        
        env = Environment("mi-entorno")
        env.python.conda_dependencies = CondaDependencies.create(
            conda_packages=['pandas', 'scikit-learn', 'numpy'],
            pip_packages=['xgboost', 'azureml-defaults']
        )
        
        # Se requiere source_directory="." para que el contenedor suba todo el repositorio 
        # y pueda resolver 'from src.zorrouno import processor'
        inf_config = InferenceConfig(
            entry_script="API/score.py", 
            source_directory=".", 
            environment=env
        )
        aci_config = AciWebservice.deploy_configuration(cpu_cores=0.5, memory_gb=0.5)
        
        service = Model.deploy(workspace=self.ws, name="api-service", 
                               models=[model], inference_config=inf_config, 
                               deployment_config=aci_config, overwrite=True)
        
        service.wait_for_deployment(show_output=True)
        return service.scoring_uri
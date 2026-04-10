from azureml.core import Workspace, Environment, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.conda_dependencies import CondaDependencies

class AzureDeployer:
    def __init__(self, subscription_id, resource_group, workspace_name):
        self.ws = Workspace.get(name=workspace_name, 
                                subscription_id=subscription_id, 
                                resource_group=resource_group)

    def register_and_deploy(self, model_path, model_name):
        # 1. Registrar
        model = Model.register(model_path=model_path, model_name=model_name, workspace=self.ws)
        
        # 2. Configurar Entorno (Tus requirements.txt van aquí)
        env = Environment("mi-entorno")
        env.python.conda_dependencies = CondaDependencies.create(
            conda_packages=['pandas', 'scikit-learn', 'numpy'],
            pip_packages=['xgboost', 'azureml-defaults']
        )
        
        # 3. Desplegar
        inf_config = InferenceConfig(entry_script="score.py", environment=env)
        aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
        
        service = Model.deploy(workspace=self.ws, name="api-service", 
                               models=[model], inference_config=inf_config, 
                               deployment_config=aci_config, overwrite=True)
        
        service.wait_for_deployment(show_output=True)
        return service.scoring_uri
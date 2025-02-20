from src.config.configuration import Configuration
from src.components.model.model import ModelIngestion


class ModelPipeline():
    def __init__(self):

        configuration = Configuration()
        self.modelconfig = configuration.model_config()

    def run_model_creating(self):

        model_create = ModelIngestion(self.modelconfig)
        model_create.create_model_and_save()


if __name__=="__main__":
    
    model_pipeline=ModelPipeline()
    model_pipeline.run_model_creating()
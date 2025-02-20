from src.config.configuration import Configuration
from src.components.data_transformation.data_transformation import DataTransformation

class DataTransformationPipeline():
    def __init__(self):
        
        configuration=Configuration()
        self.datatransformationconfig=configuration.data_transformation_config()
    

    def run_data_transformation(self):
    
        data_transformation=DataTransformation(self.datatransformationconfig)
        data_transformation.initiate_transformation()
        
        
if __name__=="__main__":
    
    data_transformation_pipeline=DataTransformationPipeline()
    data_transformation_pipeline.run_data_transformation()
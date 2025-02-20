from src.config.configuration import Configuration
from src.components.testing.testing import Testing


class TestPipeline():
    def __init__(self):

        configuration = Configuration()
        self.test_config = configuration.test_config()

    def run_testing(self):

        testing = Testing(self.test_config)
        test_result=testing.initiate_testing()
        return test_result


if __name__=="__main__":
    
    test_pipeline=TestPipeline()
    test_result=test_pipeline.run_testing()
    



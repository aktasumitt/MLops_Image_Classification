from src.config.configuration import Configuration
from src.components.prediction.predict import Prediction

class PredictionPipeline():
    def __init__(self):

        configuration=Configuration()
        self.prediction_config=configuration.prediction_config()
    
    def run_prediction_pipeline(self):
    
        prediction=Prediction(self.prediction_config)
        predict_results=prediction.predict_and_save_result()
        
        return predict_results
        
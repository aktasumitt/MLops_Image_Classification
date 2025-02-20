from torch.utils.data import DataLoader
from src.entity.config_entity import PredictionConfig
from src.components.prediction.predict_module import predict
from src.components.prediction.load_data import PreprocessPredictionData
from src.utils import save_as_json,load_obj

from src.logging.logger import logger
from src.exception.exception import ExceptionNetwork,sys
from pathlib import Path


class Prediction():
    
    def __init__(self,config:PredictionConfig):
        self.config=config
        self.model=load_obj(path=config.final_model_path).to(config.device)
        

    def load_dataset(self):
        try:
            preprocess_data=PreprocessPredictionData(resize_size=self.config.image_size)                    
            prediction_dataset=preprocess_data.data_transformation(self.config.predict_data_path)
            prediction_dataloader=DataLoader(dataset=prediction_dataset,batch_size=self.config.batch_size,shuffle=False)

            return prediction_dataloader
        except Exception as e:
           raise ExceptionNetwork(e,sys)
    
    def initiate_prediction(self):
        try:
            predict_dataloader=self.load_dataset()            
            
            prediction_labels=predict(prediction_dataloader=predict_dataloader,
                                        Model=self.model,
                                        devices=self.config.device)
            
            
            return prediction_labels
        
        except Exception as e:
           raise ExceptionNetwork(e,sys)
    
    def predict_and_save_result(self):
        try:
            label_names=self.config.labels
            label_names={v:k for k,v in label_names.items()}
            
            prediction_labels=self.initiate_prediction()
            image_paths=Path(self.config.predict_data_path).glob("*")

            results={}
            for i,img_path in enumerate(image_paths):
                label=prediction_labels[i]
                results[str(img_path)]=label_names[(label)]
                
            save_as_json(results,save_path=self.config.save_prediction_result_path)
            logger.info(f"Prediction results is saved on [ {self.config.save_prediction_result_path} ]")
            
            return list(results.values())    
             
        except Exception as e:
           raise ExceptionNetwork(e,sys)    
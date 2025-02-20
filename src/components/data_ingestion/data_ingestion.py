import zipfile
import os
from PIL import Image
from pathlib import Path
from src.exception.exception import ExceptionNetwork,sys
from src.logging.logger import logger
from src.entity.config_entity import DataIngestionConfig


class DataIngestion():
    def __init__(self,config:DataIngestionConfig,TEST_MODE:bool=False):
        self.config=config
        self.TEST_MODE=TEST_MODE
        
    # extract zip file
    def data_ingestion(self):
        
        try:
            data_location_path=(self.config.EXAMPLE_DATA_FOR_PYTEST if self.TEST_MODE ==True else self.config.data_location_path)
            save_path = Path(self.config.all_data_save_path)
            local_path = Path(data_location_path)

            os.makedirs(save_path, exist_ok=True) # create directory

            
            with zipfile.ZipFile(local_path, "r") as zip_ref:
                zip_ref.extractall(save_path)
                
            logger.info(f"zip file was extracked on [{save_path} ] from [{local_path} ]")   
        except Exception as e:
            raise ExceptionNetwork(e,sys)
    
    
    def data_separating(self):
        
        try:
            os.makedirs(self.config.train_data_path, exist_ok=True)
            os.makedirs(self.config.test_data_path, exist_ok=True)
            os.makedirs(self.config.valid_data_path, exist_ok=True)

            for dataset in Path(self.config.all_data_save_path).glob("*"):
                for label in dataset.glob("*"):  

                    self.label_train_path = os.path.join(self.config.train_data_path, label.name)
                    self.label_test_path = os.path.join(self.config.test_data_path, label.name)
                    self.label_valid_path = os.path.join(self.config.valid_data_path, label.name)
                    
                    os.makedirs(self.label_train_path, exist_ok=True)
                    os.makedirs(self.label_test_path, exist_ok=True)
                    os.makedirs(self.label_valid_path, exist_ok=True)

                    image_paths = list(label.glob("*"))
                    length = len(image_paths)
                    test_len = int(self.config.test_split_rate * length)
                    valid_len = int(self.config.valid_split_rate * length)

                    for idx, img_path in enumerate(image_paths):
                        img = Image.open(img_path)

                        if idx < test_len:
                            img.save(os.path.join(self.label_test_path, img_path.name))
                        
                        elif idx < (test_len+valid_len):
                            img.save(os.path.join(self.label_valid_path, img_path.name))
                        
                        else:
                            img.save(os.path.join(self.label_train_path, img_path.name))
            logger.info(f"Train, test and valid folders were created and images were saved on them") 
                 
        except Exception as e:
            raise ExceptionNetwork(e,sys)
        

    def initialize_data_ingestion(self):
        try:
            
            self.data_ingestion()
            self.data_separating()
        except Exception as e:
            raise ExceptionNetwork(e,sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initialize_data_ingestion()

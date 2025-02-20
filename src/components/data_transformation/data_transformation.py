from src.components.data_transformation.dataset import CreateDataset
from torchvision.transforms import transforms
from src.utils import save_obj
from src.exception.exception import ExceptionNetwork, sys
from src.entity.config_entity import DataTransformationConfig
from src.logging.logger import logger
from pathlib import Path


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_transformer(self):
        try:
            # base transforms for all stages
            base_transforms = [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Resize((self.config.img_resize_size, self.config.img_resize_size)),
                transforms.Grayscale(self.config.channel_size)
                ]
            
            # train test ve valid transformer
            train_transformer = transforms.Compose(base_transforms)
            valid_transformer = transforms.Compose(base_transforms)
            test_transformer = transforms.Compose(base_transforms)
            return train_transformer, valid_transformer, test_transformer
        
        except Exception as e:
            raise ExceptionNetwork(e, sys)

    
    def get_img_label_dict(self,path):
        img_label_dict={}
        for img_folder in Path(path).glob("*"):
            label=self.config.labels[img_folder.name]
            for img in img_folder.glob("*"):
                img_label_dict[img]=label
        return img_label_dict
    
    
    def create_and_save_dataset(self, data_path, transformer, save_path):
        try:
            # create dataset and save
            img_label_dict=self.get_img_label_dict(data_path)
            dataset = CreateDataset(img_label_dict=img_label_dict, transformer=transformer)
            save_obj(dataset, save_path)
            
        except Exception as e:
            raise ExceptionNetwork(e, sys)


    def initiate_transformation(self):
        try:
            
            # Transformers
            train_transformer, valid_transformer, test_transformer = self.get_transformer()
            
            # Creatin datasets and save
            self.create_and_save_dataset(self.config.train_data_path, train_transformer, self.config.transformed_train_dataset)
            logger.info(f"train dataset was saved on [{self.config.transformed_train_dataset} ] ")
            
            self.create_and_save_dataset(self.config.test_data_path, test_transformer, self.config.transformed_test_dataset)
            logger.info(f"test dataset was saved on [{self.config.transformed_test_dataset} ] ")
            
            self.create_and_save_dataset(self.config.valid_data_path, valid_transformer, self.config.transformed_valid_dataset)
            logger.info(f"valid dataset was saved on [{self.config.transformed_valid_dataset} ] ")
            

        except Exception as e:
            raise ExceptionNetwork(e, sys)



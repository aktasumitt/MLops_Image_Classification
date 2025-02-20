from src.components.data_transformation.dataset import CreateDataset
from torchvision.transforms import transforms
from pathlib import Path
from src.exception.exception import ExceptionNetwork, sys


class PreprocessPredictionData():
    def __init__(self, resize_size):

        self.prediction_transformer = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Resize((resize_size, resize_size)),
                                                         transforms.Normalize((0.5,), (0.5,)),
                                                         transforms.Grayscale(1)])
        
    def get_img_label_dict(self, path):
        try:
            img_label_dict = {}
            for img_files in Path(path).glob("*"):
                label = "_"
                img_label_dict[str(img_files)] = label
            return img_label_dict

        except Exception as e:
            raise ExceptionNetwork(e, sys)
           
    def data_transformation(self,data_path):
        try:
            img_path_dict=self.get_img_label_dict(data_path)
            transformed_data = CreateDataset(img_label_dict=img_path_dict, transformer=self.prediction_transformer)
            return transformed_data

        except Exception as e:
            raise ExceptionNetwork(e, sys)

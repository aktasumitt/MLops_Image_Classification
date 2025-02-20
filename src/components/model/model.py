import torch.nn as nn
from src.utils import save_obj
from src.exception.exception import ExceptionNetwork, sys
from src.entity.config_entity import ModelConfig
from src.logging.logger import logger


class CNN_Model(nn.Module):
    def __init__(self, channel_size: int, label_size: int, img_size: int):
        super(CNN_Model, self).__init__()

        self.channel_size = channel_size
        self.label_size = label_size
        self.down_block1 = self.down_block(channel_size, 64, 3, 1)
        self.down_block2 = self.down_block(64, 128, 2, 1)
        self.down_block3 = self.down_block(128, 256, 3, 1)

        self.last_conv = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
                                       nn.BatchNorm2d((256)),
                                       nn.ReLU())

        self.conv_size = ((int(img_size/8)-2)**2)*256

        self.linear1 = self.linear_block(self.conv_size, 1024)
        self.linear2 = self.linear_block(1024, 128)
        self.linear_last = nn.Linear(128, label_size)

    def down_block(self, in_channels, out_channels, kernel_size, padding, padding_mode="reflect"):
        try:
            return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode),
                                 nn.BatchNorm2d((out_channels)),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.MaxPool2d(kernel_size=2, stride=2))    # (img_s/2)
        except Exception as e:
            raise ExceptionNetwork(e, sys)

    def linear_block(self, in_features, out_features):
        try:

            return nn.Sequential(nn.Linear(in_features=in_features, out_features=out_features),
                                 nn.ReLU())
        except Exception as e:
            raise ExceptionNetwork(e, sys)

    def forward(self, data):
        try:
            x = self.down_block1(data)
            x = self.down_block2(x)
            x = self.down_block3(x)
            x = self.last_conv(x)

            x = x.view(-1, self.conv_size)

            x = self.linear1(x)
            x = self.linear2(x)
            out = self.linear_last(x)
            return out

        except Exception as e:
            raise ExceptionNetwork(e, sys)





class ModelIngestion():
    def __init__(self,config:ModelConfig):
        self.config=config
        self.model=CNN_Model(self.config.channel_size,
                             label_size=self.config.label_size, 
                             img_size=self.config.img_size)
    
    def create_model_and_save(self):
        save_obj(self.model, self.config.model_save_path)
        logger.info(f"model was saved in [{self.config.model_save_path} ]")

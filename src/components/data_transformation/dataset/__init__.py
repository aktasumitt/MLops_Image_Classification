from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL import Image

class CreateDataset(Dataset):
    def __init__(self,img_label_dict:dict,transformer:transforms):
        super(Dataset,self).__init__()
        self.img_label_dict=img_label_dict
        self.transformer=transformer
                
    def __len__(self):
        return len(list(self.img_label_dict.items()))
    
    
    def __getitem__(self, index):
        image_path,label=list(self.img_label_dict.items())[index]
        image=Image.open(image_path)
        image_transformed=self.transformer(image)
        
        return (image_transformed,label)
                
            
        
        
        
        
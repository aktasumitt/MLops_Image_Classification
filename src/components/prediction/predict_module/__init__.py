import torch
from src.exception.exception import ExceptionNetwork,sys


def predict(prediction_dataloader, Model, devices="cpu"):
    Model.eval()
    try:
        prediction_list=[]
        for _b, (img_test, _l) in enumerate(prediction_dataloader):

            img_test = img_test.to(devices)
            
            out_test = Model(img_test)    

            _, predictions_valid = torch.max(out_test, 1)
            
            prediction_list.append(predictions_valid)
        
        return torch.stack(prediction_list).reshape(-1).tolist()
    
    except Exception as e:
        raise ExceptionNetwork(e,sys)
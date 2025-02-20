from src.components.training.model_train import model_training
from src.components.training.model_valid import model_validation

from src.utils import save_as_json,load_obj,load_checkpoints,save_checkpoints,save_obj
from torch.utils.data import DataLoader
import torch
from src.logging.logger import logger
from src.exception.exception import ExceptionNetwork,sys
import mlflow
from mlflow.models import infer_signature
import subprocess
from src.entity.config_entity import TrainingConfig

import dagshub
dagshub.init(repo_owner='umitaktas', repo_name='Mlops_Classification_Project', mlflow=True)


class Training():
    
    def __init__(self,config:TrainingConfig,TEST_MODE:bool=False):
        self.config=config
        self.TEST_MODE=TEST_MODE
        self.model=load_obj(self.config.model_path).to(self.config.device)
        
        self.optimizer=torch.optim.Adam(self.model.parameters(),
                                        lr=self.config.learning_rate,
                                        betas=(self.config.beta1,self.config.beta2))
        
        self.lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=1, factor=0.5)

        
    def load_object(self):
        try:
            train_dataset=load_obj(self.config.train_dataset_path)
            train_dataloader=DataLoader(train_dataset,batch_size=self.config.batch_size,shuffle=True)
            
            valid_dataset=load_obj(self.config.validation_dataset_path)
            valid_dataloader=DataLoader(valid_dataset,batch_size=self.config.batch_size,shuffle=False)
            
            loss_fn=torch.nn.CrossEntropyLoss()        
            
            return train_dataloader,valid_dataloader,loss_fn
        
        except Exception as e:
            ExceptionNetwork(e,sys)
    
    def load_checkpoints(self,load):
        starting_epoch=1
        if load==True:
            starting_epoch=load_checkpoints(path=self.config.checkpoint_path,model=self.model,optimizer=self.optimizer)
            logger.info(f"Checkpoints were loaded. Training is starting from {starting_epoch}.epoch")
        return starting_epoch
    
    def initiate_training(self):
        try:
            result_list=[]
            
            train_dataloader,valid_dataloader,loss_fn=self.load_object()
            starting_epoch=self.load_checkpoints(load=self.config.load_checkpoint)
            
            epochs=self.config.epochs
            for epoch in range(starting_epoch,epochs):
                    
                train_loss, train_acc = model_training(train_dataloader=train_dataloader,
                                                        optimizer=self.optimizer,
                                                        loss_fn=loss_fn,
                                                        Model=self.model,
                                                        device=self.config.device)
                    
                valid_loss,valid_acc = model_validation(valid_dataloader=valid_dataloader,
                                                        loss_fn=loss_fn,
                                                        Model=self.model,
                                                        device=self.config.device)
                
                self.lr_scheduler.step(valid_loss)
                    
                save_checkpoints(save_path=self.config.checkpoint_path,model=self.model,optimizer=self.optimizer,epoch=epoch)
                logger.info(f"The last checkpoints was saved on [{self.config.checkpoint_path} ] for {epoch}.epoch")
                    
                metrics={"train_loss":train_loss,
                         "train_acc":train_acc,
                         "valid_loss": valid_loss,
                         "valid_acc":valid_acc,
                         "Epoch":epoch}
                    
                # save the metrics to the list
                result_list.append(metrics)

                # save the metrics to the mlflow
                mlflow.log_metrics(metrics=metrics,step=epoch)
                
            # Create an mlflow signature
            img_batch=next(iter(train_dataloader))[0].to("cuda")
            signature=infer_signature(model_input=img_batch.cpu().detach().numpy(),
                                          model_output=self.model(img_batch).cpu().detach().numpy())
            # save results
            save_as_json(data=result_list,save_path=self.config.save_result_path)
            logger.info(f"Training results were saved as json file on [{self.config.save_result_path} ]")
                
            # save final model
            save_obj(self.model,save_path=self.config.final_model_save_path)
            logger.info(f"Final model is saved on [{self.config.final_model_save_path}]")
        
            return signature
        
        except Exception as e:
            ExceptionNetwork(e,sys)
        
    def start_training_with_mlflow(self):
        
        try:
            
           # uri for mlflow track url in dagshub or local host
            uri="https://dagshub.com/umitaktas/Mlops_Classification_Project.mlflow"   # for dagshub
          
            # mlflow ui and other apps dont overlap
            # subprocess.Popen(["mlflow","ui"])
            
            # MLFLOW tracking
            mlflow.set_tracking_uri(uri=uri)
            logger.info(f"MLflow was tracked on [{uri} ]")
            

            # create a new MLFLOW experiment            
            mlflow.set_experiment("MLFLOW MyFirstExperiment")

            params={"Batch_size":self.config.batch_size,
                    "Learning_rate":self.config.learning_rate,
                    "Betas":(self.config.beta1,self.config.beta2),
                    "Epoch":self.config.epochs}
            
            # start an MLFLOW run
            with mlflow.start_run():

                # log the hyperparameters (epoch,lr,vs)
                mlflow.log_params(params=params)
                
                # Set a tag that we can use to remind ourselves what this run was for
                mlflow.set_tag("Pytorch Training Info","Environment image classification training")
                
                # Training
                training_artifacts,signature=self.initiate_training()
                
                # log the model
                mlflow.pytorch.log_model(
                    pytorch_model=self.model,
                    artifact_path="FirstModel",
                    signature=signature,
                    registered_model_name="my_model_name")
                
                logger.info("Training is completed. Metrics, parameters and model was saved on MLflow")
                
            return training_artifacts
        
        except Exception as e:
            ExceptionNetwork(e,sys)
            
                     
                        
            
        
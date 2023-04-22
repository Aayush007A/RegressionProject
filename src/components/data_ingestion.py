import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import save_object

# Initialization of the data ingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')
    
# Class for data ingestion

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')
        try:
            df = pd.read_csv('notebooks/data/train.csv')
            logging.info('Dataset converted into Dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            logging.info('Train test split started')
            train_set,test_set = train_test_split(df,test_size=0.3,random_state=7)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,
                             header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,
                             header=True)
            logging.info('Train test split completed')
            logging.info('Ingestion step is completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
            
        except Exception as e:
            logging.error('Exception occured at Data Ingestion step')
            raise CustomException(e,sys)
        
# Run Data Ingestion
# if __name__ == "__main__":
    
#     # Creating object of Data ingestion class
#     obj = DataIngestion()
    
#     # Obtaining train and test data path using initiate_data_ingestion method
#     train_data_path,test_data_path = obj.initiate_data_ingestion()
    
#     # Creating object of Data Transformation class
#     data_transformation = DataTransformation()
    
#     # Obtaining train ,test array and preprocessor object path 
#     train_arr,test_arr,preprocessor_obj_path = data_transformation.initiate_data_transformation(train_path=train_data_path, test_path=test_data_path)
    
#     # Creating object of Model Trainer class
#     model_obj = ModelTrainer()
    
#     # Obtaining model.pkl file and best model
#     model_obj.initate_model_training(train_array=train_arr, test_array=test_arr) 
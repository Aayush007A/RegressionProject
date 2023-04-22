
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    
    # Creating object of Data ingestion class
    obj = DataIngestion()
    
    # Obtaining train and test data path using initiate_data_ingestion method
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    
    # Creating object of Data Transformation class
    data_transformation = DataTransformation()
    
    # Obtaining train ,test array and preprocessor object path 
    train_arr,test_arr,preprocessor_obj_path = data_transformation.initiate_data_transformation(train_path=train_data_path, test_path=test_data_path)
    
    # Creating object of Model Trainer class
    model_obj = ModelTrainer()
    
    # Obtaining model.pkl file and best model
    model_obj.initate_model_training(train_array=train_arr, test_array=test_arr) 



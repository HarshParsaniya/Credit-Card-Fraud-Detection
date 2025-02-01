import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_card_fraud_detection.exception.exception import CustomException

import mlflow



@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts', 'raw.csv')
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            with mlflow.start_run(run_name='Data Ingestion'):
                # Read the Original csv file
                data_path = os.path.join('notebook/dataset/creditcard_main.csv')
                df = pd.read_csv(data_path)

                mlflow.log_param('Original_Data_Shape', df.shape)

                train, test = train_test_split(df, test_size=0.2, random_state=42)

                mlflow.log_param('Train_Rows', train.shape[0])
                mlflow.log_param('Train_Columns', train.shape[1])
                mlflow.log_param('Test_Rows', test.shape[0])
                mlflow.log_param('Test_Columns', test.shape[1])

                os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

                # Store the train and test dataset
                df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)
                train.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
                test.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

                mlflow.log_param('All_files_created', 'successfully')
                mlflow.log_param('Data_Ingestion', 1)

                return (
                    self.data_ingestion_config.train_data_path,
                    self.data_ingestion_config.test_data_path
                )


        except Exception as e:
            mlflow.log_param('Data_Ingestion', 0)
            raise CustomException(e, sys)
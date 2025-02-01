import os
import sys

from credit_card_fraud_detection.components.data_ingestion import DataIngestion
from credit_card_fraud_detection.components.data_transformation import DataTransformation
from credit_card_fraud_detection.components.model_trainer import ModelTrainer


if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_array, test_array)
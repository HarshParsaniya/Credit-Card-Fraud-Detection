import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from credit_card_fraud_detection.exception.exception import CustomException
from credit_card_fraud_detection.utils.utils import save_object, evaluate_model

import mlflow



@dataclass
class ModelTrainerConfig:
    logisticregression_data_path = os.path.join('artifacts', 'logisticregression.pkl')
    decisiontree_data_path = os.path.join('artifacts', 'decisiontree.pkl')
    supportvectormachine_data_path = os.path.join('artifacts', 'svc.pkl')
    randomforest_data_path = os.path.join('artifacts', 'randomforest.pkl')
    knearestneighbors_data_path = os.path.join('artifacts', 'knn.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            with mlflow.start_run(run_name='Model Trainer'):
                # Split train and test arrays
                X_train, X_test, y_train, y_test = (
                    train_array[:, :-1],
                    test_array[:, :-1],
                    train_array[:, -1],
                    test_array[:, -1],
                )

                mlflow.log_param('Training_Row_shape', X_train.shape[0])
                mlflow.log_param('Training_Column_shape', X_train.shape[1])
                mlflow.log_param('Testing_Row_shape', X_test.shape[0])
                mlflow.log_param('Testing_Column_shape', X_test.shape[1])

                # Model list 
                models = {
                    'logisticregression' : LogisticRegression(),
                    'decisiontree' : DecisionTreeClassifier(),
                    'randomforest' : RandomForestClassifier(),
                    'supportvectormachine' : SVC(probability=True),
                    'knearestneighbors' : KNeighborsClassifier(),
                }

                all_model_report = {}

                for model_name, model in models.items():
                    model.fit(X_train, y_train)
                    accuracy_report = evaluate_model(X_train, X_test, y_train, y_test, model)
                    all_model_report[model_name] = accuracy_report

                    # Save the model
                    model_path = getattr(self.model_trainer_config, f"{model_name}_data_path")
                    save_object(model_path, model)
                    mlflow.log_param(f"{model_name}_model_path", model_path)

                # Log accuracy scores
                mlflow.log_dict(all_model_report, "model_accuracy.json")

                mlflow.log_param('Accuracy_score', all_model_report)
         
                mlflow.log_param('Model_Trainer', 1)


        except Exception as e:
            mlflow.log_param('Model_Trainer', 0)
            raise CustomException(e, sys)
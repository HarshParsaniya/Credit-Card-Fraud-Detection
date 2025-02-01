import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from credit_card_fraud_detection.exception.exception import CustomException
from credit_card_fraud_detection.utils.utils import save_object

import mlflow



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.target_column = 'Class'

    def get_data_transformation_obj(self):
        try:
            under = RandomUnderSampler(sampling_strategy=0.1)
            smote = SMOTE(sampling_strategy=0.5)

            hybrid_pipeline = Pipeline([
                ('under', under),
                ('smote', smote)
            ])

            return hybrid_pipeline


        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            with mlflow.start_run(run_name='Data Transformation'):
                # Read the train and test file
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                # Get the data transformation object
                preprocessor = self.get_data_transformation_obj()

                # Separate features and target variable
                X_train = train_df.drop(self.target_column, axis=1)
                y_train = train_df[self.target_column]

                X_test = test_df.drop(self.target_column, axis=1)
                y_test = test_df[self.target_column]

                # Apply the data transformation to the train and test data
                X_train_transformed, y_train_resampled = preprocessor.fit_resample(X_train, y_train)

                # Final train and test data
                train_array = np.c_[X_train_transformed, y_train_resampled]
                test_array = np.c_[X_test, y_test]
                
                # Data store in pickle file
                save_object(
                    self.data_transformation_config.preprocessor_obj_file_path,
                    preprocessor
                )
                # Log artifacts in MLflow
                mlflow.log_artifact(self.data_transformation_config.preprocessor_obj_file_path, artifact_path='preprocessor')

                mlflow.log_param('Data_Transformation', 1)

                return(
                    train_array,
                    test_array,
                    self.data_transformation_config.preprocessor_obj_file_path
                )



        except Exception as e:
            mlflow.log_param('Data_Transformation', 0)
            raise CustomException(e, sys)
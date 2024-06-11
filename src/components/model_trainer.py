import sys
import os

import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('splitting Independent and Dependent variables from Train and Test data')
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            logging.info('Creating Training model')
            models={
                'LinearRegration':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                'DecisionTree':DecisionTreeRegressor(),
                'RandomForest':RandomForestRegressor()
            }

            logging.info('Model evaluation start')
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            logging.info(f'Model Report : {model_report}')
            logging.info('\n====================================================================================\n')
            
            logging.info('getting best model score from dictionary')
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            logging.info('\n====================================================================================\n')

            logging.info('saving of model.pkl has started')
            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                obj=best_model
            )

            logging.info('model.pkl is created and saved sucessfully')

        except Exception as e:
            logging.info('Exception occured in fn: initiate_model_training')
            raise CustomException(e,sys)
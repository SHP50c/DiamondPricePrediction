import os
import sys
import pickle

import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        logging.info('save_object fn start')
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info('Exception occured in fn: save object')
        raise CustomException(e,sys)


def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        logging.info('evaluate_model fn start')
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            logging.info('Train Model')
            model.fit(X_train,y_train)

            logging.info('Test Model')
            y_pred=model.predict(X_test)

            logging.info('Getting R2 score')
            test_model_score=r2_score(y_test,y_pred)

            report[list(models.keys())[i]]=test_model_score
        return report
    
    except Exception as e:
        logging.info('Exception occured fn:evaluate_model')
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        logging.info('preprocessor.pkl and model.pkl is being loaded')
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occured fn:load_object')
        raise CustomException(e,sys)
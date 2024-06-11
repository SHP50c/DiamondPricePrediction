import os
import sys

import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    ##Note:-features are derived from app.py
    def predict(self,features):
        try:
            logging.info('Predictionpipeline starts')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            logging.info(f'prediction was sucessfull\n predicted price of diamond is:{pred}')
            return pred
        
        except Exception as e:
            logging.info('Exception occured in fn:predict')
            raise CustomException(e,sys)
        

class CustomData:
    logging.info('CustomData class iscalled by app.py data is being collected by  the user')
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut=cut
        self.color=color
        self.clarity=clarity

    def get_data_as_dataframe(self):
        try:
            logging.info('Data collected by user is now being converted to DataFrame')
            custom_data_input_dict={
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df=pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame created')
            return df
        except Exception as e:
            logging.info('Exception occured in fn:get_data_as_dataframe')
            raise CustomException(e,sys)
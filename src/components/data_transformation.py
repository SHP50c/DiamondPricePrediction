import sys
import os

import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


## Data Transformation config

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')



## Data Ingestionconfig class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformation_object(self):

        try:
            logging.info('Data Transformation initiated')
            #Define Categorical and Numerical columns
            logging.info('Definning categorical and numerical column')
            categorical_cols = ['cut','color','clarity']
            numerical_cols = ['carat','depth','table','x','y','z']

            #Define custom ranking for the categories for ordinal encodding
            logging.info('Definning custom ranking for categorical_cols')
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Initiating Numerical Pipeline')
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            logging.info('Initiate Categorical Pipeline')
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )

            logging.info('Initiate Column transformation to combine both pipelines and return a preprocessor')
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
                   
            logging.info('Data transformation pipeline complete')

            return preprocessor

        except Exception as e:
            logging.info("Error in class DataTransformation (fn: get_data_transformation_object)")
            raise CustomException(e,sys)


    ## Note:- We are getting the train_data_path and test_data_path from training_pipeline.py
    def initiate_data_transformation(self,train_data_path,test_data_path):
        
        try:
            logging.info('Reading train and test data')
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info('Read train and test data sucessfully')
            logging.info(f'Train DataFrame Head:\n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head:\n{test_df.head().to_string()}')

            logging.info("Now we'll Obtain object of preprocessor from fn: get_data_transformation_object")

            preprocessing_obj=self.get_data_transformation_object()

            ##Initialising target column
            target_column_name='price'
            drop_column=[target_column_name,'id']

            logging.info('splitting the train and test data into independent and dependent features')

            input_feature_train_df=train_df.drop(columns=drop_column,axis=1)
            target_features_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_column,axis=1)
            target_features_test_df=test_df[target_column_name]

            logging.info('Applying the transformation')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info('Concatinating train and test input_features_arr with target_feature_arr')

            train_arr=np.c_[input_feature_train_arr,np.array(target_features_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_features_test_df)]

            logging.info('Saving of processor.pkl file started')
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('processor.pkl is created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info("Error occured in class Datatransformation (fn:initiate_data_transformation)")

            raise CustomException(e,sys)
            







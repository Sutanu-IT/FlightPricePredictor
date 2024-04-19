import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_function


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time',
                                'destination_city', 'class']
            numerical_cols = ['days_left']

            # Define the custom ranking for each ordinal variable
            airline_categories = ["SpiceJet", "AirAsia", "GO_FIRST", "Indigo", "Air_India", "Vistara"]
            source_city_categories = ["Chennai", "Hyderabad", "Kolkata", "Bangalore", "Mumbai", "Delhi"]
            destination_city_categories = ["Chennai", "Hyderabad", "Kolkata", "Bangalore", "Delhi", "Mumbai"]
            departure_time_categories = ["Late_Night", "Afternoon", "Night", "Evening", "Early_Morning", "Morning"]
            arrival_time_categories = ["Late_Night", "Early_Morning", "Afternoon", "Morning", "Evening", "Night"]
            class_categories = ["Business", "Economy"]
            stops_categories = ["two_or_more", "zero", "one"]
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorigal Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("OrdinalEncoder", OrdinalEncoder(categories=
                                                      [airline_categories,
                                                       source_city_categories,
                                                       departure_time_categories,
                                                       stops_categories,
                                                       arrival_time_categories,
                                                       destination_city_categories,
                                                       class_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_cols),
                    ('cat_pipeline', cat_pipeline, categorical_cols)
                ]
            )

            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name, 'Unnamed: 0','flight']

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            ## Trnasformating using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_function(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.info("Exception occurred in the initiate datatransformation")

            raise CustomException(e, sys)

import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj
import pandas as pd


class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info("Error occurred in predict function in prediction_pipeline location")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, days_left: float,
                 class_type: str,
                 airline: str,
                 arrival_time: str,
                 destination_city: str,
                 stops: str,
                 source_city: str,
                 departure_time: str):
        self.airline = airline
        self.source_city = source_city
        self.departure_time = departure_time
        self.stops = stops
        self.arrival_time = arrival_time
        self.destination_city = destination_city
        self.class_type = class_type
        self.days_left = days_left

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'days_left': [self.days_left],
                'airline': [self.airline],
                'source_city': [self.source_city],
                'departure_time': [self.departure_time],
                'stops': [self.stops],
                'arrival_time': [self.arrival_time],
                'destination_city': [self.destination_city],
                'class': [self.class_type],

            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe created")
            return df
        except Exception as e:
            logging.info("Error occurred in get_data_as_dataframe function in prediction_pipeline")
            raise CustomException(e, sys)



import os
import sys
import pickle
import pandas as pd
import mysql.connector as connection
from sqlalchemy import create_engine
from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import logging


def save_function(file_path, obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def model_performance(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train models
            model.fit(X_train, y_train)
            # Test data
            y_test_pred = model.predict(X_test)
            # R2 Score
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise CustomException(e, sys)


# Function to load a particular object
def load_obj(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Error in load_object function in utils")
        raise CustomException(e, sys)


def establish_connection(host, user, password, database):
    mydb = connection.connect(host='localhost', user='root', password='', use_pure=True)
    query = 'create database diamondproject;'
    cursor = mydb.cursor()
    cursor.execute(query)


def retrieve_data(host,user_name,password,Database_name):
    df = pd.DataFrame(pd.read_csv(r"C:\Users\abhis\Documents\ThirdEyeData_local\LearningPath\Internship\Code\pipelineproject\notebooks\data\gemstone.csv"))
    engine = create_engine(f"mysql://{user_name}:{password}@{host}/{Database_name}")
    # engine = create_engine(f'mysql://{"root"}:{""}@{"localhost"}/{"diamondproject"}')
    table_name = "flightpricedata"
    try:
        df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
        print("Data entered successfully")
    except Exception as e:
        logging.info("Error in retrieve function in utils")
        raise CustomException(e, sys)
    query = f"SELECT * From {Database_name}"
    df1 = pd.read_sql(query, engine)

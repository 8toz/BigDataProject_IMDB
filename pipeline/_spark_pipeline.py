# PySpark Imports
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext

# PySpark ML Imports
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import Vectors
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.param import Param, Params
from pyspark.ml.feature import Bucketizer, VectorAssembler, StringIndexer

# Local Imports
from _dataset import Dataset 

# Other Imports
import pandas as pd
import duckdb
import numpy as np
import os
import sys

# System paths
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Database path
DATABASE_PATH = "./database/DDBB_duckdb.duckdb"

def input_preprocessing(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing actions required for the movies dataset.
    
    :param movies: pd.DataFrame containing the movies table from DuckDB
    """
    # Temporal fix: There are NaN columns that break the cast from Pandas Dataframe to Spark Dataframe in main().
    input_df.dropna(axis=0, inplace=True)
    input_df.loc[:, 'label'].replace({True: 'True', False: 'False'}, inplace=True)
    return input_df

def fetch_duckdb() -> list[pd.DataFrame]:
    """
    Fetches all the required data from the database and returns an array of dataframes.
    TEMP: Only data from the movies table is being fetched right now. Expand to writers
    
    :param
    """
    con = duckdb.connect(database=DATABASE_PATH, read_only=False)
    movies = con.execute('''select * from movies''').fetch_df()
    con.close()
    movies = input_preprocessing(movies)

    return [movies]

def feature_selection(data: list[pd.DataFrame]):
    """
    Feature selection function. 
    TEMP: To be done!
    
    :param list[pd.DataFrame] Containing all the required datasets (movies and writers)
    """

    return [["num_votes"]]

def generate_pipeline(features: list) -> Pipeline:
    """
    Function to generate the Spark pipeline based on the following operations:
        - Assembling (choosing) the desired features (numeric).
        - Index the selected features to be processed by the pipeline (strings).
        - Initializing the pipeline based on the indexed features.
    
    :param
    """
    assembler = VectorAssembler(inputCols=features[0], outputCol="features")
    indexer = StringIndexer(inputCol="label").setOutputCol("label-index")
    pipeline = Pipeline().setStages([assembler, indexer])
    return pipeline

def generate_output_pipeline(features: list) -> Pipeline:
    """
    Function to generate the Spark pipeline based on the following operations:
        - Assembling (choosing) the desired features (numeric).
        - Index the selected features to be processed by the pipeline (strings).
        - Initializing the pipeline based on the indexed features.
    
    :param
    """
    assembler = VectorAssembler(inputCols=features[0], outputCol="features")
    pipeline = Pipeline().setStages([assembler])
    return pipeline

def read_validation() -> pd.DataFrame:
    """
    Reads the validation dataset from its .csv file
    
    :param
    """
    validation = pd.read_csv(r"data\validation_hidden.csv")
    return validation
    
def read_test() -> pd.DataFrame:
    """
    Reads the test dataset from its .csv file
    
    :param
    """
    test = pd.read_csv(r"data\test_hidden.csv")
    return test
    
def create_submission(model, validation, test) -> None:
    """
    Create the required submission file in .csv format
    
    :param model: PySpark generated binary classifier
    """    
    pipeline = generate_output_pipeline([['numVotes']])
    pipeline_fit = pipeline.fit(validation)
    p_val = pipeline_fit.transform(validation)
    p_test = pipeline_fit.transform(test)
    
    validation = validation.toPandas()  
    test = test.toPandas()  
    
    validation["label"] = model.transform(p_val).select('prediction').collect().tolist
    test["label"] = model.transform(p_test).select('prediction').collect().tolist

    # Cast to bool and store in .csv
    validation["label"].astype(bool).to_csv("val_result.csv", index=False, header=None)
    test["label"].astype(bool).to_csv("test_result.csv", index=False, header=None)

    # Generate final submission
    for file in ["val_result.csv", "test_result.csv"]:
        with open(file, 'r+') as f:
            f.seek(0,2)                    
            size=f.tell()               
            f.truncate(size-2)
    
def main() -> None:
    """
    Main PySpark pipeline execution.
    
    :param
    """
    # Initialize PySpark Context
    conf = SparkConf().setAppName("binary-ml-classification")
    sc = SparkContext.getOrCreate(conf)
    sqlContext = SparkSession.builder.getOrCreate()
    
    # Fetch data and process features to obtain a Spark Dataframe
    data = fetch_duckdb()
    features = feature_selection(data)
    df_spark = sqlContext.createDataFrame(data[0])
    
    # Generate the pipeline
    pipeline = generate_pipeline(features)
    
    # Fit the pipeline using the Spark Dataframe
    pipeline_fit = pipeline.fit(df_spark)  
    
    # Generate and train the model
    prepared = pipeline_fit.transform(df_spark)
    dt = DecisionTreeClassifier(labelCol = "label-index", featuresCol= "features")
    dt_model = dt.fit(prepared)
    
    # Read output generation files
    validation = read_validation()
    test = read_test()
    
    df_validation = sqlContext.createDataFrame(validation)
    df_test = sqlContext.createDataFrame(test)

    create_submission(dt_model, df_validation, df_test)
    


if __name__ == '__main__':
    main()
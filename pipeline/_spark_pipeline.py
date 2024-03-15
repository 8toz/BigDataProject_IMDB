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


def fetch_duckdb() -> list[pd.DataFrame]:
    """
    Fetches all the required data from the database and returns an array of dataframes.
    TEMP: Only data from the movies table is being fetched right now. Expand to writers
    
    :param
    """
    con = duckdb.connect(database=DATABASE_PATH, read_only=False)
    movies = con.execute('''
    WITH director_avg_scores AS (
        SELECT 
            d.director_id,
            COALESCE(SUM(CASE WHEN m.label THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 0.5) AS director_avg_score
        FROM 
            directing d
        INNER JOIN 
            movies m ON d.movie_id = m.movie_id
        WHERE 
            m.subset = 'train'
        GROUP BY 
            d.director_id
    ),
    director_scores AS (
        SELECT 
            d.movie_id,
            COUNT(d.director_id) AS director_count,
            AVG(COALESCE(das.director_avg_score, 0.5)) AS director_avg_score
        FROM 
            directing d
        LEFT JOIN 
            director_avg_scores das ON das.director_id = d.director_id
        GROUP BY 
            d.movie_id
    ),
    writer_avg_scores AS (
        SELECT 
            w.writer_id,
            COALESCE(SUM(CASE WHEN m.label THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 0.5) AS writer_avg_score
        FROM 
            writing w
        INNER JOIN 
            movies m ON w.movie_id = m.movie_id
        WHERE 
            m.subset = 'train'
        GROUP BY 
            w.writer_id
    ),
    writer_scores AS (
        SELECT 
            w.movie_id,
            COUNT(w.writer_id) AS writer_count,
            AVG(COALESCE(was.writer_avg_score, 0.5)) AS writer_avg_score
        FROM 
            writing w
        LEFT JOIN 
            writer_avg_scores was ON w.writer_id = was.writer_id
        GROUP BY 
            w.movie_id
    )
    SELECT 
        m.movie_id,
        m.num_votes,
        m.runtime_min,
        m.title_length,
        ds.director_avg_score,
        COALESCE(ds.director_count, 0) AS director_count,
        m.label,
        ws.writer_avg_score,
        COALESCE(ws.writer_count, 0) AS writer_count
    FROM 
        movies m
    LEFT JOIN 
        director_scores ds ON m.movie_id = ds.movie_id
    LEFT JOIN 
        writer_scores ws ON m.movie_id = ws.movie_id
    WHERE 
        m.subset = 'train';
    ''').fetch_df()
    con.close()

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

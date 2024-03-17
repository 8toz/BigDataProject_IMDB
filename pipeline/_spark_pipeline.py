# PySpark Imports
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql.session import SparkSession
from pyspark.sql import DataFrame

# PySpark ML Imports
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Parsing and requests Imports
import requests
from bs4 import BeautifulSoup

# Other Imports
import pandas as pd
import duckdb
import os
import sys

# System paths
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Database path
DATABASE_PATH = "../database/DDBB_duckdb.duckdb"

# Competition URL paths
login_url = 'http://big-data-competitions.swedencentral.cloudapp.azure.com:8080/auth/login'
upload_url = 'http://big-data-competitions.swedencentral.cloudapp.azure.com:8080/competitions/imdb/submit'
submissions_url = 'http://big-data-competitions.swedencentral.cloudapp.azure.com:8080/submissions/'


# Credentials for authentication
username = 'group25'
password = '6XwJgJRh'

# Path to your val_result.csv and test_result.csv files
val_csv_path = 'val_result.csv'
test_csv_path = 'test_result.csv'


def fetch_duckdb() -> list[pd.DataFrame]:
    """
    Fetches all the required data from the database and returns an array of dataframes.
    TEMP: Only data from the movies table is being fetched right now. Expand to writers
    
    :param
    """
    con = duckdb.connect(database=DATABASE_PATH, read_only=False)
    df = con.execute('''
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
    ),
    numbered_movies AS (
        SELECT
            m.*,
            ROW_NUMBER() OVER () AS row_num
        FROM 
            movies m
    )
    SELECT
        nm.subset, 
        nm.movie_id,
        nm.num_votes,
        nm.runtime_min,
        nm.title_length,
        COALESCE(ds.director_avg_score, 0.5) AS director_avg_score,
        COALESCE(ds.director_count, 0) AS director_count,
        CASE WHEN nm.label THEN 1 ELSE 0 END AS label,
        nm.label AS label_og,
        COALESCE(ws.writer_avg_score, 0.5) AS writer_avg_score,
        COALESCE(ws.writer_count, 0) AS writer_count
    FROM 
        numbered_movies nm
    LEFT JOIN 
        director_scores ds ON nm.movie_id = ds.movie_id
    LEFT JOIN 
        writer_scores ws ON nm.movie_id = ws.movie_id
    ORDER BY
        nm.row_num;
    ''').fetch_df()
    con.close()
    
    
    train = df[df['subset'] == 'train'].drop(['subset'], axis=1).dropna()
    test = df[df['subset'] == 'test'].drop(['subset', 'label'], axis=1)
    validation = df[df['subset'] == 'val'].drop(['subset', 'label'], axis=1)
    
    return train, test, validation

def generate_pipeline(features: list) -> Pipeline:
    """
    Function to generate the Spark pipeline based on the following operations:
        - Assembling (choosing) the desired features (numeric).
        - Index the selected features to be processed by the pipeline (strings).
        - Initializing the pipeline based on the indexed features.
    
    :param
    """
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    pipeline = Pipeline().setStages([assembler])
    return pipeline

def generate_output_pipeline(features: list) -> Pipeline:
    """
    Function to generate the Spark pipeline based on the following operations:
        - Assembling (choosing) the desired features (numeric).
        - Index the selected features to be processed by the pipeline (strings).
        - Initializing the pipeline based on the indexed features.
    
    :param
    """
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    pipeline = Pipeline().setStages([assembler])
    return pipeline

def hyper_parameter_tuning(prepared: DataFrame) -> None:
    """
    Function to find the best hyperparameters for a RandomForestClassifier
    
    :param
    """
    
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    
    # Define parameter grid for hyperparameter tuning
    paramGrid = (ParamGridBuilder()
                 .addGrid(rf.numTrees, [100, 200, 300])  # Number of trees in the forest
                 .addGrid(rf.maxDepth, [2, 5, 10, 15])    # Maximum depth of each tree
                 .addGrid(rf.maxBins, [5, 10, 20, 32])
                 .build())
    
    # Define evaluator
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    
    # Define cross-validation
    cv = CrossValidator(estimator=rf,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=5)  # Use 5 folds
    
    # Train model using cross-validation
    cv_model = cv.fit(prepared)
    
    # Best model from cross-validation
    best_model = cv_model.bestModel
    
    best_max_depth = best_model._java_obj.getMaxDepth()
    best_num_trees = best_model._java_obj.getNumTrees()
    best_max_bins = best_model._java_obj.getMaxBins()
    print("Best maxDepth:", best_max_depth)
    print("Best numTrees:", best_num_trees)
    print("Best maxBins:", best_max_bins)
    
def create_submission(model, validation, test, features) -> None:
    """
    Create the required submission file in .csv format
    
    :param model: PySpark generated binary classifier
    """    
    pipeline = generate_output_pipeline(features)
    pipeline_fit = pipeline.fit(validation)
    p_val = pipeline_fit.transform(validation)
    p_test = pipeline_fit.transform(test)
    
    val_results = model.transform(p_val).select('prediction').toPandas()
    test_results = model.transform(p_test).select('prediction').toPandas()

    # Cast to bool and store in .csv
    val_results.astype(bool).to_csv("val_result.csv", index=False, header=None)
    test_results.astype(bool).to_csv("test_result.csv", index=False, header=None)
    
def automated_submission_online() -> None:
    """
    Uploads the CSV files to the web server and prints the most recent score and the best score
    
    :param
    """
    # Create a session
    session = requests.Session()

    # Login to the website
    login_data = {
        'username': username,
        'password': password
    }

    login_response = session.post(login_url, data=login_data)

    # Check if login was successful (you may need to adjust this based on the website's response)
    if login_response.status_code == 200:
        pass
    else:
        print("Login failed. Status code:", login_response.status_code)
        exit()

    # Create a dictionary containing the files to be uploaded
    files = {
        'valid': open(val_csv_path, 'rb'),
        'test': open(test_csv_path, 'rb')
    }

    # Make the POST request to upload the files
    upload_response = session.post(upload_url, files=files)

    # Check if the upload was successful
    if upload_response.status_code == 200:
        # Parse the HTML response
        soup = BeautifulSoup(upload_response.text, 'html.parser')
        # Find the submission made by "group25"
        submission_rows = soup.find_all('tr', class_='submission')
        # Iterate through each submission row
        for row in submission_rows:
            # Find the <th> tag with scope="row" and text equal to "group25" within the current row
            submission_group = row.find('th', scope='row', string='group25')
            
            # Check if the submission_group is found in the current row
            if submission_group:
                # Extract the submission time and validation score
                submission_time = row.find('td').get_text()
                validation_score = submission_group.find_next_sibling('td').get_text()
                
                # Print the most recent submission for "group25"
                print(f"Best submission by group25: Time: {submission_time}, Validation Score: {validation_score}")
                # Once we find the submission, we can break out of the loop
                break
        else:
            # If no submission for "group25" is found in any of the rows
            print("Submission by group25 not found.")
            
        # Send a GET request to the submissions page
        submissions_response = session.get(submissions_url)
        
        # Check if the request was successful
        if submissions_response.status_code == 200:
            # Parse the HTML response
            soup = BeautifulSoup(submissions_response.text, 'html.parser')
            # Find all submission rows
            submission_rows = soup.find_all('tr', class_='submission')
            if submission_rows:
                # Extract information from the most recent submission
                most_recent_submission = submission_rows[0]  # The first row is the most recent one
                submission_time = most_recent_submission.find('td')
                validation_score = submission_time.find_next_sibling('td')
                print(f"Most recent submission by group25: Time: {submission_time.get_text()}, Validation Score: {validation_score.get_text()}")
            else:
                print("No submissions found.")   
        else:
            print("Submission by group25 not found.")
    else:
        print("Failed to upload files. Status code:", upload_response.status_code)

    
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
    train, test, validation = fetch_duckdb()
    features = ["runtime_min", "num_votes", "director_avg_score", "director_count", "writer_avg_score", "writer_count"]
    df_train = sqlContext.createDataFrame(train)
    
    # Generate the pipeline
    pipeline = generate_pipeline(features)
    
    # Fit the pipeline using the Spark Dataframe
    pipeline_fit = pipeline.fit(df_train)  
    
    # Generate and train the model
    prepared = pipeline_fit.transform(df_train)

    # Run if need to tune hyperparameters
    # hyper_parameter_tuning(prepared)
    
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth = 10, numTrees=300, maxBins=20)
    best_model = rf.fit(prepared)
    
    
    # # Read output generation files
    df_validation = sqlContext.createDataFrame(validation)
    df_test = sqlContext.createDataFrame(test)

    create_submission(best_model, df_validation, df_test, features)
    automated_submission_online()
    
    
if __name__ == "__main__":
    main()

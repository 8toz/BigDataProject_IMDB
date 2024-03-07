import pandas as pd
import numpy as np
from datetime import datetime


def integrate_movies(train_df, validation_df, test_df):
     # We label the rows to know from which subset the data comes from 
    train_df["subset"] = "train"
    validation_df["subset"] = "val"
    validation_df["label"] = np.nan
    test_df["subset"] = "test"
    test_df["label"] = np.nan

    return pd.concat([train_df, validation_df, test_df], axis=0)

def preprocess_movies(train_df, validation_df, test_df):
    movies_df = integrate_movies(train_df, validation_df, test_df)

    # NaN value correction
    mask = movies_df["startYear"].isna()
    movies_df.loc[mask, "startYear"] = movies_df.loc[mask, "endYear"]
    movies_df = movies_df.drop(columns="endYear")

    movies_df["numVotes"] = movies_df["numVotes"].fillna(0) 
    movies_df["runtimeMinutes"] = movies_df["runtimeMinutes"].fillna(0) 


    movies_df["startYear"] = movies_df["startYear"].astype(int)
    movies_df["numVotes"] = movies_df["numVotes"].astype(int)
    movies_df["runtimeMinutes"] = movies_df["runtimeMinutes"].astype(int)


    return movies_df


#movie_id	primary_title	original_title	start_year	runtime_min	num_votes	label	subset	audit_time
def merge_movies(df, con):

        con.register('movies_stg', df)
        
        con.execute(''' INSERT INTO movies (movie_id, primary_title, original_title, start_year, runtime_min, num_votes, label, subset)
                        SELECT tconst as movie_id, 
                               primaryTitle as primary_title,
                               originalTitle as original_title, 
                               startYear as start_year, 
                               runtimeMinutes as runtime_min, 
                               numVotes as num_votes, 
                               label as label, 
                               subset as subset
                        FROM movies_stg
                    ''')

import pandas as pd
import numpy as np


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
    mask = movies_df[movies_df["startYear"].isna()].index
    # The end date column does not tell us anything so we drop it and fill start date with it
    movies_df.loc[mask, "startYear"] = movies_df.loc[mask, "endYear"]
    movies_df = movies_df.drop(columns="endYear")

    return movies_df

def merge_movies(df, con):
    try:
        con.register('movies_stg', df)
        con.execute('''INSERT INTO movies SELECT * FROM movies_stg; ''')
    except Exception as e:
        print("Error:", str(e))
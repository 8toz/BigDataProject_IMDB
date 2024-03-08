import pandas as pd
import numpy as np
from datetime import datetime
from unidecode import unidecode


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

    movies_df["numVotes"] = movies_df["numVotes"].fillna(movies_df["numVotes"].median()) 
    movies_df["runtimeMinutes"] = movies_df["runtimeMinutes"].fillna(movies_df["runtimeMinutes"].median()) 

    movies_df["startYear"] = movies_df["startYear"].astype(int)
    movies_df["numVotes"] = movies_df["numVotes"].astype(int)
    movies_df["runtimeMinutes"] = movies_df["runtimeMinutes"].astype(int)

    movies_df = _process_title_data(movies_df)

    return movies_df


def _process_title_data(df: pd.DataFrame) -> pd.DataFrame:
    """We check whether the primary/ original tiles are different."""
    transtab = str.maketrans(dict.fromkeys('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~ |', ' '))

    df["primaryTitle"] = df["primaryTitle"].apply(lambda x: unidecode(str(x).lower().translate(transtab)))
    df["originalTitle"] = df["originalTitle"].apply(lambda x: unidecode(str(x).lower().translate(transtab)))

    df["title_changed"] = df.apply(lambda row: (not row["primaryTitle"] == row["originalTitle"]) and (row["originalTitle"] is not None), axis=1)
    df["title_length"] = df.apply(lambda row: len(row["primaryTitle"]), axis=1)
    return df


#movie_id	primary_title	original_title	start_year	runtime_min	num_votes	label	subset	audit_time
def merge_movies(df, con):

        con.register('movies_stg', df)
        
        con.execute(''' INSERT INTO movies (movie_id, primary_title, original_title, start_year, runtime_min, num_votes, label, subset, title_changed, title_length)
                        SELECT tconst as movie_id, 
                               primaryTitle as primary_title,
                               originalTitle as original_title, 
                               startYear as start_year, 
                               runtimeMinutes as runtime_min, 
                               numVotes as num_votes, 
                               label as label, 
                               subset as subset,
                               title_changed as title_changed,
                               title_length as title_length
                        FROM movies_stg ON CONFLICT (movie_id) DO UPDATE 
                            SET primary_title = EXCLUDED.primary_title,
                                original_title = EXCLUDED.original_title,
                                start_year = EXCLUDED.start_year,
                                runtime_min = EXCLUDED.runtime_min,
                                num_votes = EXCLUDED.num_votes,
                                label = EXCLUDED.label,
                                subset = EXCLUDED.subset
                    ''')

import pandas as pd
import duckdb

STAGING_PATH = "./data"
DATABASE_PATH = "./database/DDBB_duckdb.duckdb"

def preprocess_directors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the directors data by dropping NaN values and duplicates

    df - Directors DataFrame
    returns - Cleaned Directors DataFrame
    """
    df.replace('\\N', pd.NA, inplace=True)
    df.dropna(axis=0, inplace=True)
    df.drop_duplicates(inplace=True)
    
    return df

def merge_directors(df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> None:
    """
    Inserts the data into the directors table. ON CONFLICT will mean that we are inserting duplicates
    that is why we set the DO NOTHING command (in this table all columns form the primary key)
    """

    con.register('directing_stg', df)
    con.execute('''INSERT INTO  directing (director_id, movie_id) 
                    SELECT director as director_id,
                            movie as movie_id
                    FROM directing_stg ON CONFLICT (director_id, movie_id) DO NOTHING
                ''')
    return None

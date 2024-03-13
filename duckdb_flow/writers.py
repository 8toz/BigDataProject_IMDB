import pandas as pd
import duckdb

def preprocess_writers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the writers data by dropping NaN values and duplicates

    df - Writers DataFrame
    returns - A cleaned writers DataFrame
    """
    df.replace('\\N', pd.NA, inplace=True)
    df.dropna(axis=0, inplace=True)
    df.drop_duplicates(inplace=True)
    
    return df

def merge_writers(df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> None:    
    """
    Cleans the directors data by dropping NaN values and duplicates

    df - Directors DataFrame
    returns - Cleaned Directors DataFrame
    """
    con.register('writing_stg', df)
    con.execute('''INSERT INTO writing (writer_id, movie_id) 
                    SELECT writer as writer_id, 
                            movie as movie_id,
                    FROM writing_stg ON CONFLICT (writer_id, movie_id) DO NOTHING; 
                ''')
    return None

       
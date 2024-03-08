import pandas as pd

STAGING_PATH = "./data"
DATABASE_PATH = "./database/DDBB_duckdb.duckdb"

def preprocess_directors(df):
    df.replace('\\N', pd.NA, inplace=True)
    df.dropna(axis=0, inplace=True)
    df.drop_duplicates(inplace=True)
    
    return df

def merge_directors(df, con):

    con.register('directing_stg', df)
    con.execute('''INSERT INTO  directing (director_id, movie_id) 
                    SELECT director as director_id,
                            movie as movie_id
                    FROM directing_stg ON CONFLICT (director_id, movie_id) DO NOTHING
                ''')
 

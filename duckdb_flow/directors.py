import pandas as pd

STAGING_PATH = "./data"
DATABASE_PATH = "./database/DDBB_duckdb.duckdb"

def preprocess_directors(df):
    df.replace('\\N', pd.NA, inplace=True)
    df.dropna(axis=0, inplace=True)
    df.drop_duplicates(inplace=True)
    
    return df

def merge_directors(df, con):
    try:
        con.register('directing_stg', df)
        con.execute('''INSERT INTO directing SELECT * FROM directing_stg; ''')
    except Exception as e:
        print("Error:", str(e))
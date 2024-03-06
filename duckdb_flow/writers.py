import pandas as pd

def preprocess_writers(df):
    df.replace('\\N', pd.NA, inplace=True)
    df.dropna(axis=0, inplace=True)
    df.drop_duplicates(inplace=True)
    
    return df

def merge_writers(df, con): 
    try:
        con.register('writing_stg', df)
        con.execute('''INSERT INTO writing SELECT * FROM writing_stg; ''')
    except Exception as e:
        print("Error:", str(e))
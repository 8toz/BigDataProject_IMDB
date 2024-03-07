import pandas as pd

def preprocess_writers(df):
    df.replace('\\N', pd.NA, inplace=True)
    df.dropna(axis=0, inplace=True)
    df.drop_duplicates(inplace=True)
    
    return df

def merge_writers(df, con): 
        
        con.register('writing_stg', df)
        con.execute('''INSERT INTO writing (writer_id, movie_id) 
                        SELECT writer as writer_id, 
                               movie as movie_id,
                        FROM writing_stg; ''')


       
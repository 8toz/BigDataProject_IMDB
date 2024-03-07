import pandas as pd
import duckdb
import os

from duckdb_flow.movies import preprocess_movies, merge_movies
from duckdb_flow.directors import preprocess_directors, merge_directors
from duckdb_flow.writers import preprocess_writers, merge_writers

STAGING_PATH = "./data"
DATABASE_PATH = "./database/DDBB_duckdb.duckdb"

def get_processed_files():
    con = duckdb.connect(database=DATABASE_PATH, read_only=False)
    processed_files = []
    result = con.execute('''select * from processed_files''')
    rows = result.fetchall()
    [processed_files.append(row[0]) for row in rows]
    con.close()

    return processed_files

def is_valid(file):
    return True if ".csv" in file or ".json" in file else False


def preprocess_data(processed_files):
    # Checks for unprocessed files
    trigger = 0
    dfs = []
    new_files = []
    directing_df = pd.DataFrame()
    writing_df = pd.DataFrame()
    for file in os.listdir(STAGING_PATH):
        if file not in processed_files and is_valid(file):
            trigger += 1
            new_files.append(file)
            fn_no_ext = file.split(".")[0]
            if ".json" in file:
                globals()[fn_no_ext+"_df"] = pd.read_json(os.path.join(STAGING_PATH, file))
                print("Created ", fn_no_ext+"_df", " dataframe.")
            elif ".csv" in file:
                if "validation" in file:
                    validation_df = pd.read_csv(os.path.join(STAGING_PATH, file), index_col=0, na_values=['\\N'])
                    continue
                elif "test" in file:
                    test_df = pd.read_csv(os.path.join(STAGING_PATH, file), index_col=0, na_values=['\\N'])
                    continue
                df = pd.read_csv(os.path.join(STAGING_PATH, file), index_col=0, na_values=['\\N'])
                print("Appending: ", df.shape[0], " rows...")
                dfs.append(df)
                train_df = pd.concat(dfs, ignore_index=True)
            else: 
                pass
        if trigger == 0:
            print("All files processed")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    directing_df = globals()['directing_df']
    writing_df = globals()['writing_df']    
    

    movies_df = preprocess_movies(train_df, validation_df, test_df)
    directing_df= preprocess_directors(directing_df)
    writing_df = preprocess_writers(writing_df)


    con = duckdb.connect(database=DATABASE_PATH, read_only=False)

    merge_movies(movies_df, con)
    merge_directors(directing_df, con)
    merge_writers(writing_df, con)

    con.register('files_checked', pd.DataFrame(new_files))
    con.execute('''INSERT INTO processed_files SELECT * FROM files_checked; ''')
    con.close()

    return movies_df, directing_df, writing_df
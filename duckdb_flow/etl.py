import pandas as pd
import duckdb
import os

from duckdb_flow.movies import preprocess_movies, merge_movies
from duckdb_flow.directors import preprocess_directors, merge_directors
from duckdb_flow.writers import preprocess_writers, merge_writers

STAGING_PATH = "./data"
DATABASE_PATH = "./database/DDBB_duckdb.duckdb"

def get_processed_files() -> list:
    """
    Return - a list of the processed files to avoid processing them again
    """
    con = duckdb.connect(database=DATABASE_PATH, read_only=False)
    processed_files = []
    result = con.execute('''select * from processed_files''')
    rows = result.fetchall()
    [processed_files.append(row[0]) for row in rows]
    con.close()

    return processed_files

def is_valid(file: str) -> bool:
    """
    Check wether the files are csv or json ignore the rest
    """
    return True if ".csv" in file or ".json" in file else False


def preprocess_data(processed_files: list) -> pd.DataFrame:
    """
    Main method that inserts the data into the SQL tables and orquestrate everything
    1 - We check if files have not been processed yet
    2 - We extract the csv data and the json one into separated files
    3 - We integrate and clean it
    4 - We update the tables with the new data
    5 - We insert the new files in the processed_files tables to avoid reading through them again

    Returns - Movies, directing and writers DataFrames if any (This method can be improved)
    """
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
                if "validation_hidden.csv" in file:
                    validation_df = pd.read_csv(os.path.join(STAGING_PATH, file), index_col=0, na_values=['\\N'])
                    continue
                elif "test_hidden.csv" in file:
                    test_df = pd.read_csv(os.path.join(STAGING_PATH, file), index_col=0, na_values=['\\N'])
                    continue
                elif "train" in file:
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

    con.register('files_checked', pd.DataFrame(new_files, columns=["file_name"]))
    con.execute('''INSERT INTO processed_files (file_name)
                    SELECT file_name as file_name
                    FROM files_checked; ''')
    con.close()

    return movies_df, directing_df, writing_df
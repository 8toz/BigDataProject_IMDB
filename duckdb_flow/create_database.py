import duckdb

STAGING_PATH = "./data"
DATABASE_PATH = "./database/DDBB_duckdb.duckdb"

def create_database() -> str:

    """
    Creates all the tables for the duckdb DDBB is they do not exist
    """

    con = duckdb.connect(database=DATABASE_PATH, read_only=False)

    con.execute('''
    CREATE TABLE IF NOT EXISTS movies (
        movie_id char(10) NOT NULL, 
        primary_title varchar(100) default NULL, 
        original_title varchar(100) default NULL, 
        start_year integer default NULL,
        runtime_min integer default NULL, 
        num_votes integer default NULL,
        label boolean default NULL,
        subset char(5) default NULL,
        title_changed boolean default NULL,
        title_length integer default NULL,
        audit_time timestamp default CURRENT_TIMESTAMP,
        
        PRIMARY KEY (movie_id) 
        );
                
                
    CREATE TABLE IF NOT EXISTS directing (
        director_id char(10) NOT NULL, 
        movie_id char(10) default NULL,
        audit_time timestamp default CURRENT_TIMESTAMP,
        
        PRIMARY KEY (director_id, movie_id),
        FOREIGN KEY (movie_id) REFERENCES movies(movie_id));
                
    CREATE TABLE IF NOT EXISTS writing (
        writer_id char(10) NOT NULL, 
        movie_id char(10) default NULL,
        audit_time timestamp default CURRENT_TIMESTAMP, 
        
        PRIMARY KEY (writer_id, movie_id), 
        FOREIGN KEY (movie_id) REFERENCES movies(movie_id));
                
    CREATE TABLE IF NOT EXISTS processed_files (
        file_name varchar(200),
        audit_time timestamp default CURRENT_TIMESTAMP,
        
        PRIMARY KEY (file_name)
        );
    ''')

    con.close()

    return "Database Created"
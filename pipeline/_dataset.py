import pandas as pd
from unidecode import unidecode


class Dataset:
    _splits: tuple[float, float]

    _complete_df: pd.DataFrame
    _director_scores: pd.DataFrame
    _writer_scores: pd.DataFrame

    _directing: pd.DataFrame
    _writing: pd.DataFrame

    _train_df: pd.DataFrame | None
    _test_df: pd.DataFrame | None

    def __init__(self, splits: tuple[float, float]) -> None:
        """
        Initialize the Dataset.

        :param splits: The sizes for Train / Test split.
        """
        assert sum(splits) == 1., "CAUTION: the splits dont sum up to 1!"
        self._test_df = None
        self._train_df = None
        self._splits = splits

        self._make_initial_dataframes()
        self._complete_df = self.process_dataframe(df=self._complete_df)

    def _make_initial_dataframes(self) -> None:
        """Populate the complete DataFrame."""
        self._directing = pd.read_json("../data/directing.json")
        self._writing = pd.read_json("../data/writing.json")

        main_df = pd.concat([pd.read_csv(f"../data/train-{i}.csv") for i in range(1, 9)])

        self._complete_df = self._merge_df_with_auxiliary_data(main_df)
        self._directing_scores = self._complete_df.groupby("director").agg({"label": ["mean", "sum", "count"]})
        self._writer_scores = self._complete_df.groupby("writer").agg({"label": ["mean", "sum", "count"]})

    def _merge_df_with_auxiliary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the dataframe with external data sources.

        :param df: The dataframe.
        :return: The merged dataframe.
        """
        df = df.merge(self._directing, left_on="tconst", right_on="movie")
        df = df.merge(self._writing, left_on="tconst", right_on="movie")
        return df

    @staticmethod
    def _initial_df_cleaning(df: pd.DataFrame) -> pd.DataFrame:
        """Initial cleaning of the data."""
        df.set_index("tconst", inplace=True)
        for elem in ["Unnamed: 0", "movie_x", "movie_y"]:
            try:
                df.drop(elem, axis=1, inplace=True)  # Here we remove unwanted columns
            except KeyError:
                pass

        df.fillna({"numVotes": df["numVotes"].median()}, inplace=True)  # We fill nan values with median of votes.
        df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")
        df.fillna({"runtimeMinutes": df["runtimeMinutes"].median()}, inplace=True)
        return df

    def _format_writer_director_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Here we aggregate the writers and directors of the movies into lists.
        Additionally, we use the directors and writers to extract scores for the movies.
        """
        def __make_unique_list(elem) -> list:
            return list(set(list(elem)))

        def __keep(elem):
            return list(elem)[0]

        def __get_scores(elem, target: pd.DataFrame) -> tuple[float, float, float]:

            total, mean, count = 0., 0., 0.
            for uid in elem:
                try:
                    total += target[("label", "sum")].loc[uid]
                    mean += target[("label", "mean")].loc[uid]
                    count += target[("label", "count")].loc[uid]
                except KeyError:  # If a uid is not in the external data sources.
                    print(f"Error: key not found {uid}")
                    total += target[("label", "sum")].mean()
                    mean += target[("label", "mean")].mean()
                    count += target[("label", "count")].mean()
            return mean / len(elem), total / len(elem), total-count

        process_cols = ["director", "writer"]
        to_keep = list(set(df.columns) - set(process_cols))
        modify_dict = {elem: __make_unique_list for elem in process_cols} | {elem: __keep for elem in to_keep}

        df = df.groupby(df.index, as_index=False).agg(modify_dict).reset_index()
        df[["director_score_mean_mean", "director_score_mean_total", "director_score_total_pos"]] = df[
            "director"].apply(lambda elem: pd.Series(__get_scores(elem, target=self._directing_scores)))
        df[["writer_score_mean_mean", "writer_score_mean_total", "writer_score_total_pos"]] = df["writer"].apply(
            lambda elem: pd.Series(__get_scores(elem, target=self._writer_scores)))
        df.drop(process_cols, axis=1, inplace=True)
        return df

    @staticmethod
    def _format_year_data(df: pd.DataFrame) -> pd.DataFrame:
        """We combine year data, since they seem to be exclusive."""
        year_cols = ["endYear", "startYear"]
        for col in year_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df["year"] = df["endYear"].fillna(df["startYear"])
        df["votes_per_year"] = df["numVotes"] / (2024 - df["year"])
        df.drop(year_cols, axis=1, inplace=True)
        return df

    @staticmethod
    def _process_title_data(df: pd.DataFrame) -> pd.DataFrame:
        """We check whether the primary/ original tiles are different."""
        transtab = str.maketrans(dict.fromkeys('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~ |', ''))

        df["primaryTitle"] = df["primaryTitle"].apply(lambda x: unidecode(str(x).lower().translate(transtab)))
        df["originalTitle"] = df["originalTitle"].apply(lambda x: unidecode(str(x).lower().translate(transtab)))

        df["titles_changed"] = df.apply(lambda row: (not row["primaryTitle"] == row["originalTitle"]) and (row["originalTitle"] is not None), axis=1)
        df["title_len"] = df.apply(lambda row: len(row["primaryTitle"]), axis=1)
        return df

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the complete df for feature extraction and other operations.

        :param df: The dataframe to process.
        :return: The processed df.
        """
        if any([col not in df.columns for col in ["writer", "director"]]):  # check if all external data is merged
            df = self._merge_df_with_auxiliary_data(df)
        df = self._initial_df_cleaning(df)
        df = self._format_writer_director_data(df)
        df = self._format_year_data(df)
        df = self._process_title_data(df)
        return df

    def generate_datasplit(self, careful: bool = True) -> None:
        """
        Generate a new data-split from the data.

        :param careful: If careful, then we don't generate a new datasplit if there is already train and test splits available.
        :raises ValueError: If there is already a data-split and we are in careful mode.
        """

        if careful and self._train_df is not None:
            raise ValueError("There is already a split available, please use careful=False to overwrite.")

        df = self._complete_df.sample(frac=1)
        train_size, test_size = tuple([int(z * df.size) for z in self._splits])

        self._train_df = df.head(train_size)
        self._test_df = df.tail(test_size)

    @property
    def train_df(self) -> pd.DataFrame:
        """
        Return the train df.

        :return: The df.
        :raises ValueError: If no train df is found.
        """
        if self._train_df is not None:
            return self._train_df
        raise ValueError("No train df generated yet. Please use generate_datasplit()")

    @property
    def test_df(self) -> pd.DataFrame:
        """
        Return the test df.

        :return: The df.
        :raises ValueError: If no test df is found.
        """
        if self._test_df is not None:
            return self._test_df
        raise ValueError("No test df generated yet. Please use generate_datasplit()")

    @property
    def complete_df(self) -> pd.DataFrame:
        """
        Return the complete df.

        :return: The df.
        """
        return self._complete_df

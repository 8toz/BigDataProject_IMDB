import pandas as pd
from unidecode import unidecode


class Dataset:
    _splits: tuple[float, float]

    _complete_df: pd.DataFrame
    _director_scores: pd.DataFrame
    _writer_scores: pd.DataFrame
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

        self._make_dataframes()
        self._complete_df = self._process_dataframe(df=self._complete_df)

    def _make_dataframes(self) -> None:
        """Populate the complete DataFrame."""
        directing = pd.read_json("../data/directing.json")
        writing = pd.read_json("../data/writing.json")

        main_df = pd.concat([pd.read_csv(f"../data/train-{i}.csv") for i in range(1, 9)])

        main_df = main_df.merge(directing, left_on="tconst", right_on="movie")
        self._complete_df = main_df.merge(writing, left_on="tconst", right_on="movie")
        self._directing_scores = self._complete_df.groupby("director").agg({"label": ["mean", "sum"]})
        self._writer_scores = self._complete_df.groupby("writer").agg({"label": ["mean", "sum"]})

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the complete df for feature extraction and other operations.

        :param df: The dataframe to process.
        :return: The processed df.
        """
        df.set_index("tconst", inplace=True)
        df.drop(["Unnamed: 0", "movie_x", "movie_y"], axis=1,
                inplace=True)  # Here we remove unwanted columns
        df["numVotes"].fillna(df["numVotes"].median(), inplace=True)  # We fill nan values with median of votes.
        df["runtimeMinutes"].replace({r"\N": None}, inplace=True)
        df["runtimeMinutes"].fillna(df["runtimeMinutes"].median(), inplace=True)

        """We aggregate the writers and directors of the movies into lists."""

        def __make_unique_list(elem):
            return list(set(list(elem)))

        process_cols = ["director", "writer"]
        to_keep = list(set(df.columns) - set(process_cols))
        df = df.groupby(to_keep, as_index=False).agg(
            {elem: __make_unique_list for elem in process_cols}).reset_index()

        """Now we use the directors and writers to extract scores for the movies."""

        def __get_scores(elem, target: pd.DataFrame) -> tuple[float, float]:
            total, mean = 0., 0.
            for uid in elem:
                total += target[("label", "sum")].loc[uid]
                mean += target[("label", "mean")].loc[uid]
            return mean / len(elem), total / len(elem)

        df[["director_score_mean_mean", "director_score_mean_total"]] = df[
            "director"].apply(lambda elem: pd.Series(__get_scores(elem, target=self._directing_scores)))
        df[["writer_score_mean_mean", "writer_score_mean_total"]] = df["writer"].apply(
            lambda elem: pd.Series(__get_scores(elem, target=self._writer_scores)))
        df.drop(process_cols, axis=1, inplace=True)

        """We combine year data, since they seem to be exclusive-"""
        year_cols = ["endYear", "startYear"]
        df[year_cols] = df[year_cols].replace(r"\N", None)
        df["year"] = df["endYear"].fillna(df["startYear"])
        df.drop(year_cols, axis=1, inplace=True)

        """We check whether the primary/ original tiles are different."""
        transtab = str.maketrans(dict.fromkeys('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~ |', ''))

        def __check_titles(row) -> bool:
            """Check if titles are different."""
            pt = unidecode(row["primaryTitle"].lower().translate(transtab))
            ot = unidecode(row["originalTitle"].lower().translate(transtab))
            return not pt == ot

        df["titles_changed"] = df.apply(__check_titles, axis=1)
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

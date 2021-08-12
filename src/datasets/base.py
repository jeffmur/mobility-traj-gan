"""src/datasets/base.py

A base class for mobility datasets.
"""
import abc
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


from src import config

LOG = logging.getLogger(__name__)


def stratified_split(
    df: pd.DataFrame,
    label_column: str,
    trajectory_column: str,
    test_size: float = 0.2,
    min_trajectories: int = 10,
):
    """Split the dataset into train and test sets, stratifying trajectories by label.

    A stratified split on label to balance label classes across the
    test and train sets. Random assignment is at the group (trajectory
    ID) level.  Labels with less than 2 trajectories will be dropped
    so that each label shows up at least once in each of the train and
    test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame containing the trajectory data to split.
    label_column: str
        The name of the column containing the label to stratify by.
    trajectory_column: str
        The name of the column containing the trajectory ID to group assign by.
    test_size : float
        The ratio of the data that should be assigned to the test set. Default is 20%.
    min_trajectories : int
        The minimum number of trajectories a subject must have in order to be included.
    """
    ids = df[[label_column, trajectory_column]].drop_duplicates()
    ids = ids.groupby("label").filter(lambda x: len(x) >= min_trajectories)
    split = StratifiedShuffleSplit(test_size=test_size, n_splits=2, random_state=config.SEED).split(
        ids, ids[label_column]
    )
    train_tids, test_tids = next(split)
    train_set = df[df[trajectory_column].isin(train_tids)]
    test_set = df[df[trajectory_column].isin(test_tids)]
    return train_set, test_set


class Dataset(abc.ABC):
    """Base class for mobility datasets."""

    label_column: str = "label"
    trajectory_column: str = "tid"
    datetime_column: str = "datetime"
    lat_column: str = "lat"
    lon_column: str = "lon"

    @abc.abstractmethod
    def preprocess(self) -> pd.DataFrame:
        """Preprocess the raw data."""
        raise NotImplementedError()

    def to_trajectories(
        self, *args, min_points: int = 2, resolution: str = "10min"
    ) -> pd.DataFrame:
        """Return the dataset as a Pandas DataFrame split into user-week trajectories.

        Multiple points within a `resolution` time interval will be removed (default 10 minutes).
        See https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
        for possible values.

        Parameters
        ----------
        min_points : int
            The minimum number of location points (rows) in a
            trajectory to include it in the dataset.
        args : str
            The names of any extra categorical columns to pass through.
        """

        df = self.preprocess()
        df = df.sort_values([self.label_column, self.datetime_column])
        df["year"] = df[self.datetime_column].dt.year
        df["month"] = df[self.datetime_column].dt.month
        df["day"] = df[self.datetime_column].dt.day
        df["weekday"] = df[self.datetime_column].dt.weekday
        df["hour"] = df[self.datetime_column].dt.hour
        df["minute"] = df[self.datetime_column].dt.minute
        df["week"] = df[self.datetime_column].dt.isocalendar().week
        df["timestep"] = (df[self.datetime_column].dt.floor(resolution)).astype(int)
        df = (
            df.groupby(
                [
                    self.label_column,
                    "year",
                    "month",
                    "week",
                    "day",
                    "weekday",
                    "hour",
                    "timestep",
                    *args,
                ]
            )
            .agg({self.lat_column: "mean", self.lon_column: "mean"})
            .reset_index()
        )
        # filter out trajectories with fewer than min points
        df = (
            df.groupby([self.label_column, "year", "week"])
            .filter(lambda x: len(x) >= min_points)
            .reset_index()
        )
        df[self.trajectory_column] = df.groupby([self.label_column, "year", "week"]).ngroup()
        df = df[
            [
                self.label_column,
                self.trajectory_column,
                self.lat_column,
                self.lon_column,
                "weekday",
                "hour",
                *args,
            ]
        ]

        return df

    def get_vocab_sizes(self):
        """Get a dictionary of categorical features and their cardinalities."""
        df = self.to_trajectories()
        return (
            df.drop([self.label_column, self.trajectory_column], axis=1, errors="ignore")
            .select_dtypes("int")
            .nunique()
            .to_dict()
        )

    def train_test_split(
        self,
        test_size: float = 0.2,
        min_points: int = 10,
        resolution: str = "10min",
        min_trajectories: int = 10,
    ):
        """Split the dataset into train and test sets, stratifying trajectories by label.

        A stratified split on label to balance label classes across the
        test and train sets. Random assignment is at the group (trajectory ID) level.

        Parameters
        ----------
        test_size : float
            The ratio of the data that should be assigned to the test set. Default is 20%.
        min_points: int
            The minimum number of points a single trajectory must have in order to be included.
        min_trajectories : int
            The minimum number of trajectories a subject must have in order to be included.
        """
        train_file = Path(f"data/{str(self)}_train.csv")
        test_file = Path(f"data/{str(self)}_test.csv")
        if train_file.exists() and test_file.exists():
            LOG.info("Reading train set from %s", train_file)
            LOG.info("Reading test set from %s", test_file)
            train_set = pd.read_csv(train_file)
            test_set = pd.read_csv(test_file)
            return train_set, test_set
        df = self.to_trajectories(min_points=min_points, resolution=resolution)
        train_set, test_set = stratified_split(
            df,
            self.label_column,
            self.trajectory_column,
            test_size=test_size,
            min_trajectories=min_trajectories,
        )
        LOG.info("Saving train set to %s", train_file)
        train_set.to_csv(train_file, index=False)
        LOG.info("Saving test set to %s", test_file)
        test_set.to_csv(test_file, index=False)
        return train_set, test_set

    def __repr__(self):
        return type(self).__name__

    def __str__(self):
        return type(self).__name__

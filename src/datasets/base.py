"""src/datasets/base.py

A base class for mobility datasets.
"""
import abc


class Dataset(abc.ABC):
    """Base class for mobility datasets."""

    label_column: str = "label"
    trajectory_column: str = "tid"
    datetime_column: str = "datetime"
    lat_column: str = "lat"
    lon_column: str = "lon"

    @abc.abstractmethod
    def preprocess(self):
        """Preprocess the raw data."""
        raise NotImplementedError()

    def to_trajectories(self, min_points: int = 2):
        """Return the dataset as a Pandas DataFrame split into user-week trajectories.

        Multiple points within a ten minute interval will be removed.

        Parameters
        ----------
        min_points : int
            The minimum number of location points (rows) in a
            trajectory to include it in the dataset.
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
        df["tenminute"] = (df[self.datetime_column].dt.minute // 10 * 10).astype(int)
        df = (
            df.groupby(
                [self.label_column, "year", "month", "week", "day", "weekday", "hour", "tenminute"]
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

    def __repr__(self):
        return type(self).__name__

    def __str__(self):
        return type(self).__name__

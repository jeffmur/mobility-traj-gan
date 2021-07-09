"""processors.py

Classes for pre and post processing trajectory data.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted


class GPSGridTransformer(TransformerMixin, BaseEstimator):
    """A processor that transforms GPS locations into grid positions in a rectangular grid."""


class GPSGeoHasher(TransformerMixin, BaseEstimator):
    """A processor that uses the GeoHash algorithm to discretize the GPS locations."""
    def __init__(self, precision)


class GPSNormalizer(TransformerMixin, BaseEstimator):
    """A processor that normalizes GPS coordinates as min-max scaled distance from a centroid."""

    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range)
        self.kmeans = KMeans(n_clusters=1)
        super().__init__()

    def fit(self, X):
        """Fit parameters to the data."""
        self.kmeans.fit(X)
        self.scaler.fit(X - self.kmeans.cluster_centers_[0])
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform the data."""
        check_is_fitted(self)
        center = self.kmeans.cluster_centers_[0]
        X1 = X - center
        return self.scaler.transform(X1)

    def inverse_transform(self, X):
        """Reverse the transformation."""
        X1 = self.scaler.inverse_transform(X)
        return X1 + self.kmeans.cluster_centers_[0]

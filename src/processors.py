"""processors.py

Classes for pre and post processing trajectory data.
"""

import geohash2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted


class GPSGridTransformer(TransformerMixin, BaseEstimator):
    """A processor that transforms GPS locations into grid positions in a rectangular grid."""

    # TODO


class GPSGeoHasher(TransformerMixin, BaseEstimator):
    """A processor that uses the GeoHash algorithm to discretize the GPS locations."""

    def __init__(self, precision):
        self.precision = precision
        base32 = [
            *[str(i) for i in range(0, 10)],
            *sorted(
                list(
                    set([chr(a) for a in range(ord("a"), ord("z") + 1)]) - set(["a", "i", "l", "o"])
                )
            ),
        ]
        binary = [np.asarray(list("{0:05b}".format(x)), dtype=int) for x in range(0, len(base32))]
        self.base32_to_bin = dict(zip(base32, binary))
        self.bin_to_base32 = dict(zip(["".join(list(map(str, b))) for b in binary], base32))

    def fit(self, _):
        """Fit parameters to the data."""

    def transform(self, X):
        """Transform the data."""
        hash_func = np.frompyfunc(geohash2.encode, 3, 1)
        hashed = hash_func(X[:, 0], X[:, 1], self.precision)
        to_bin = np.frompyfunc(
            lambda x: np.concatenate([self.base32_to_bin[a] for a in list(x)]), 1, 1
        )
        return np.vstack(to_bin(hashed))

    def inverse_transform(self, X):
        """Reverse the data transformation."""
        xbin = X.astype(str).reshape(X.shape[0], self.precision, 5)
        res = []
        for code in xbin:
            letters = []
            for num in code:
                letters.append(self.bin_to_base32["".join(num)])
            lat_lon = geohash2.decode_exactly("".join(letters))[0:2]
            res.append(lat_lon)
        return np.array(res)


class GPSNormalizer(TransformerMixin, BaseEstimator):
    """A processor that normalizes GPS coordinates as min-max scaled distance from a centroid."""

    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range)
        self.kmeans = KMeans(n_clusters=1)
        self.feature_range = feature_range
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

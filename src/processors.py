"""processors.py

Classes for pre and post processing trajectory data.
"""

from sklearn.base import BaseEstimator, TransformerMixin


class GPSGridTransformer(TransformerMixin, BaseEstimator):
    """A processor that transforms GPS locations into grid positions in a rectangular grid."""


class GPSGeoHasher(TransformerMixin, BaseEstimator):
    """A processor that uses the GeoHash algorithm to discretize the GPS locations."""


class GPSNormalizer(TransformerMixin, BaseEstimator):
    """A processor that normalizes GPS coordinates as min-max scaled distance from a centroid."""

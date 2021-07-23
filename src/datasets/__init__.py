"""src/datasets/__init__.py"""
from .base import Dataset
from .mdc_lausanne import MDCLausanne
from .geolife_beijing import GeoLifeBeijing
from .foursquare_nyc import FourSquareNYC

DATASETS = [MDCLausanne, GeoLifeBeijing, FourSquareNYC]

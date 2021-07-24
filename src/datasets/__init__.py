"""src/datasets/__init__.py"""
from .base import Dataset
from .mdc_lausanne import MDCLausanne
from .geolife_beijing import GeoLifeBeijing
from .foursquare_nyc import FourSquareNYC
from .privamov_lyon import PrivamovLyon

DATASETS = [MDCLausanne, GeoLifeBeijing, FourSquareNYC, PrivamovLyon]

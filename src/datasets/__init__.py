"""src/datasets/__init__.py"""
from .base import Dataset
from .mdc_lausanne import MDCLausanne
from .geolife_beijing import GeoLifeBeijing

DATASETS = [MDCLausanne, GeoLifeBeijing]

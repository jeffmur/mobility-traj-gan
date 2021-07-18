## Unit test for generating processed file 

import dotenv
from src import datasets

geo = datasets.GeoLifeBeijing()

df = geo.preprocess()

print(df.head())

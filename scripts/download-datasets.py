#!/usr/bin/env python3

import os

from sklearn.datasets import fetch_openml, fetch_20newsgroups
from folktables import ACSDataSource


DEFAULT_DATA_DIR = os.path.join("var", "data")

fetch_openml(data_id=1590, as_frame=False, data_home=DEFAULT_DATA_DIR)
fetch_20newsgroups(subset="all", data_home=DEFAULT_DATA_DIR)
ACSDataSource(survey_year="2018", horizon="1-Year", survey="person", root_dir=DEFAULT_DATA_DIR).get_data(download=True)

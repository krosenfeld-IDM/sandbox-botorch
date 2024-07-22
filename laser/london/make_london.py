"""
Make the london dataset
"""

import os
import numpy as np
import sciris as sc
from data.engwaldata import data
from collections import OrderedDict

# change directory to the parent of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

distances = np.load(os.path.join("data","engwaldist.npy"))
radius = 30 # km

# identify which locations are within 30km of London
ref_city = "London"
j = data.placenames.index(ref_city)
london_cities = OrderedDict()
for placename in data.placenames:
    i = data.placenames.index(placename)
    if distances[j, i] < radius:
        london_cities[placename] = distances[j, i]
print(f"Found {len(london_cities)} cities within {radius}km of {ref_city}.")
print("Max london_cities distance:", np.max([d for d in london_cities.values()]))
# https://github.com/krosenfeld-IDM/laser-technology-comparison/blob/ccf22b0b98dc6bb88e2f0d568e88b97542c23acb/EnglandAndWales/data/csvstopy.py#L65-L71
# construct the london dataset
londondata = sc.dictobj()
londondata.placenames = list(london_cities.keys())
londondata.years = data.years
londondata.reports = data.reports
londondata.places = OrderedDict()

print(set(london_cities.keys()) - set(londondata.placenames))

for placename in london_cities.keys():
    londondata.places[placename] = data.places[placename]

sc.save(os.path.join("data","londondata.sc"), londondata)

london_distances = np.zeros((len(london_cities), len(london_cities)))
for i, city1 in enumerate(london_cities.keys()):
    for j, city2 in enumerate(london_cities.keys()):
        london_distances[i, j] = distances[data.placenames.index(city1), data.placenames.index(city2)]
np.save(os.path.join("data","londondist.npy"), london_distances)
print("max in london_distances:", np.max(london_distances))
print("done")
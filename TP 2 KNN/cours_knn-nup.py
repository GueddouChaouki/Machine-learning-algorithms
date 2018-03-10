# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:38:15 2018

@author: pc
"""

import numpy as np
from itertools import product
from numpy.random import uniform
import seaborn as sns

sns.set(color_codes = True)

n_points = 1000
for dim in [2, 8, 32, 128, 512]:
  #lazy generation of the points (generator)
  data = (uniform(0, 1, (dim,)) for i in range(n_points))
  #numpy.linalg.norm(a-b) would be faster
  #this version is only for educational porposes
  distances = [np.sqrt(np.sum((a-b)**2)) / np.sqrt(dim)
                    for a, b in product(data, repeat=2) if a is not b]
  ax = sns.distplot(distances, kde = False, label = "D={}".format(dim))

ax.legend()
ax.set_xlabel("distance / $\sqrt{D}$")
ax.set_ylabel("# pairs of points at this distance")

fig = ax.get_figure()

fig.savefig("dist_distrib.pdf")
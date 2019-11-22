import numpy as np
import sys
import random

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec

# The following code and colors list is from vtkplotter.color : https://github.com/marcomusy/vtkplotter/blob/master/vtkplotter/colors.py
# The code is copied here just to make it easier to look up and change colros
try:
	import matplotlib
	import matplotlib.cm as cm_mpl

	_mapscales = cm_mpl
except:
	_mapscales = None
	# see below, this is dealt with in colorMap()
from matplotlib.colors import Normalize


class InvertedNormalize(Normalize):
	def __call__(self, *args, **kwargs):
		return 1 - super(InvertedNormalize, self).__call__(*args, **kwargs)

class MplColorHelper:
	"""
		Usage: instantiate the class with the CMAP to be used and the coors range. Then pass it values to get the RGB value of the color.
		"inverse" gives the possibility to invert the order of the colors in the cmap
	"""
	def __init__(self, cmap_name, start_val, stop_val, inverse=False, rgb255=False):
		self.cmap_name = cmap_name
		self.cmap = plt.get_cmap(cmap_name)
		self.rgb255 = rgb255

		if not inverse:
			self.norm = matplotlib.colors.Normalize(vmin=start_val, vmax=stop_val)
		else:
			self.norm = InvertedNormalize(vmin=start_val, vmax=stop_val)
		self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

	def get_rgb(self, val):
		if not self.rgb255:
			return self.scalarMap.to_rgba(val)[:-1]
		else:
			return [np.int(np.floor(255*c)) for c in  self.scalarMap.to_rgba(val)[:-1]]

def get_n_colors(n, cmap="tab20"):
	return [plt.get_cmap(cmap)(i) for i in np.arange(n)]


def rgb1_to_rgb255(rgb):
	return [x*255 for x in rgb]

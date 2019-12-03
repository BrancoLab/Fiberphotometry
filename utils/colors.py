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


aliceblue= "#F0F8FF"
antiquewhite= "#FAEBD7"
aqua= "#00FFFF"
aquamarine= "#7FFFD4"
azure= "#F0FFFF"
beige= "#F5F5DC"
bisque= "#FFE4C4"
blanchedalmond= "#FFEBCD"
blue= "#0000FF"
blueviolet= "#8A2BE2"
brown= "#A52A2A"
burlywood= "#DEB887"
cadetblue= "#5F9EA0"
chartreuse= "#7FFF00"
chocolate= "#D2691E"
coral= "#FF7F50"
cornflowerblue= "#6495ED"
cornsilk= "#FFF8DC"
crimson= "#DC143C"
cyan= "#00FFFF"
darkblue= "#00008B"
darkcyan= "#008B8B"
darkgoldenrod= "#B8860B"
darkgray= "#A9A9A9"
darkgreen= "#006400"
darkkhaki= "#BDB76B"
darkmagenta= "#8B008B"
darkolivegreen= "#556B2F"
darkorange= "#FF8C00"
darkorchid= "#9932CC"
darkred= "#8B0000"
darksalmon= "#E9967A"
darkseagreen= "#8FBC8F"
darkslateblue= "#483D8B"
darkslategray= "#2F4F4F"
darkturquoise= "#00CED1"
darkviolet= "#9400D3"
deeppink= "#FF1493"
deepskyblue= "#00BFFF"
dimgray= "#696969"
dodgerblue= "#1E90FF"
firebrick= "#B22222"
floralwhite= "#FFFAF0"
forestgreen= "#228B22"
fuchsia= "#FF00FF"
gainsboro= "#DCDCDC"
ghostwhite= "#F8F8FF"
gold= "#FFD700"
goldenrod= "#DAA520"
gray= "#808080"
green= "#008000"
greenyellow= "#ADFF2F"
honeydew= "#F0FFF0"
hotpink= "#FF69B4"
indianred= "#CD5C5C"
indigo= "#4B0082"
ivory= "#FFFFF0"
khaki= "#F0E68C"
lavender= "#E6E6FA"
lavenderblush= "#FFF0F5"
lawngreen= "#7CFC00"
lemonchiffon= "#FFFACD"
lightblue= "#ADD8E6"
lightcoral= "#F08080"
lightcyan= "#E0FFFF"
lightgray= "#D3D3D3"
lightgreen= "#90EE90"
lightpink= "#FFB6C1"
lightsalmon= "#FFA07A"
lightseagreen= "#20B2AA"
lightskyblue= "#87CEFA"
lightsteelblue= "#B0C4DE"
lightyellow= "#FFFFE0"
lime= "#00FF00"
limegreen= "#32CD32"
linen= "#FAF0E6"
magenta= "#FF00FF"
maroon= "#800000"
mediumaquamarine= "#66CDAA"
mediumblue= "#0000CD"
mediumorchid= "#BA55D3"
mediumpurple= "#9370DB"
mediumseagreen= "#3CB371"
mediumslateblue= "#7B68EE"
mediumspringgreen= "#00FA9A"
mediumturquoise= "#48D1CC"
mediumvioletred= "#C71585"
midnightblue= "#191970"
mintcream= "#F5FFFA"
mistyrose= "#FFE4E1"
moccasin= "#FFE4B5"
navajowhite= "#FFDEAD"
navy= "#000080"
oldlace= "#FDF5E6"
olive= "#808000"
olivedrab= "#6B8E23"
orange= "#FFA500"
orangered= "#FF4500"
orchid= "#DA70D6"
palegoldenrod= "#EEE8AA"
palegreen= "#98FB98"
paleturquoise= "#AFEEEE"
palevioletred= "#DB7093"
papayawhip= "#FFEFD5"
peachpuff= "#FFDAB9"
peru= "#CD853F"
pink= "#FFC0CB"
plum= "#DDA0DD"
powderblue= "#B0E0E6"
purple= "#800080"
rebeccapurple= "#663399"
red= "#FF0000"
rosybrown= "#BC8F8F"
royalblue= "#4169E1"
saddlebrown= "#8B4513"
salmon= "#FA8072"
sandybrown= "#F4A460"
seagreen= "#2E8B57"
seashell= "#FFF5EE"
sienna= "#A0522D"
silver= "#C0C0C0"
skyblue= "#87CEEB"
slateblue= "#6A5ACD"
slategray= "#708090"
snow= "#FFFAFA"
blackboard= "#393939"
springgreen= "#00FF7F"
steelblue= "#4682B4"
tan= "#D2B48C"
teal= "#008080"
thistle= "#D8BFD8"
tomato= "#FF6347"
turquoise= "#40E0D0"
violet= "#EE82EE"
wheat= "#F5DEB3"
white= "#FFFFFF"
whitesmoke= "#F5F5F5"
yellow= "#FFFF00"
yellowgreen= "#9ACD32"

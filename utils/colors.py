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

def _isSequence(arg):
	# Check if input is iterable.
	if hasattr(arg, "strip"):
		return False
	if hasattr(arg, "__getslice__"):
		return True
	if hasattr(arg, "__iter__"):
		return True
	return False


def getColor(rgb=None, hsv=None):
	"""
	Convert a color or list of colors to (r,g,b) format from many different input formats.
	:param bool hsv: if set to `True`, rgb is assumed as (hue, saturation, value).
	Example:
		 - RGB    = (255, 255, 255), corresponds to white
		 - rgb    = (1,1,1) is white
		 - hex    = #FFFF00 is yellow
		 - string = 'white'
		 - string = 'w' is white nickname
		 - string = 'dr' is darkred
		 - int    =  7 picks color nr. 7 in a predefined color list
		 - int    = -7 picks color nr. 7 in a different predefined list
	|colorcubes| |colorcubes.py|_
	"""
	# recursion, return a list if input is list of colors:
	if _isSequence(rgb) and (len(rgb) > 3 or _isSequence(rgb[0])):
		seqcol = []
		for sc in rgb:
			seqcol.append(getColor(sc))
		return seqcol

	if str(rgb).isdigit():
		rgb = int(rgb)

	if hsv:
		c = hsv2rgb(hsv)
	else:
		c = rgb

	if _isSequence(c):
		if c[0] <= 1 and c[1] <= 1 and c[2] <= 1:
			return c  # already rgb
		else:
			if len(c) == 3:
				return list(np.array(c) / 255.0)  # RGB
			else:
				return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, c[3])  # RGBA

	elif isinstance(c, str):  # is string
		c = c.replace("grey", "gray").replace(" ", "")

		if "#" in c:  # hex to rgb
			h = c.lstrip("#")
			rgb255 = list(int(h[i : i + 2], 16) for i in (0, 2, 4))
			rgbh = np.array(rgb255) / 255.0
			if np.sum(rgbh) > 3:
				print("Error in getColor(): Wrong hex color", c)
				return (0.5, 0.5, 0.5)
			return tuple(rgbh)
		else:
			raise ValueError()

	elif isinstance(c, int):  # color number
		if c >= 0:
			return colors1[c % 10]
		else:
			return colors2[-c % 10]

	elif isinstance(c, float):
		if c >= 0:
			return colors1[int(c) % 10]
		else:
			return colors2[int(-c) % 10]

	# print("Unknown color:", c)
	return (0.5, 0.5, 0.5)

red      = [.8, .2, .2]
blue     = [.3, .3, .9]
green    = [.2, .8, .2]
orange   = [1,  .6, .0]
pink     = [.7, .4, .5]
magenta  = [1., 0., 1.]
purple   = [.5, 0., .5]
white    = [1., 1., 1.]
black    = [0., 0., 0.]
grey     = [.7, .7, .7]
darkgrey = [.2, .2, .2]
teal     = [0., .7, .7]
lilla    = [.8, .4, .9]
lightblue = [.6, .6, .9]

aliceblue= getColor("#F0F8FF")
antiquewhite= getColor("#FAEBD7")
aqua= getColor("#00FFFF")
aquamarine= getColor("#7FFFD4")
azure= getColor("#F0FFFF")
beige= getColor("#F5F5DC")
bisque= getColor("#FFE4C4")
blanchedalmond= getColor("#FFEBCD")
blue= getColor("#0000FF")
blueviolet= getColor("#8A2BE2")
brown= getColor("#A52A2A")
burlywood= getColor("#DEB887")
cadetblue= getColor("#5F9EA0")
chartreuse= getColor("#7FFF00")
chocolate= getColor("#D2691E")
coral= getColor("#FF7F50")
cornflowerblue= getColor("#6495ED")
cornsilk= getColor("#FFF8DC")
crimson= getColor("#DC143C")
cyan= getColor("#00FFFF")
darkblue= getColor("#00008B")
darkcyan= getColor("#008B8B")
darkgoldenrod= getColor("#B8860B")
darkgray= getColor("#A9A9A9")
darkgreen= getColor("#006400")
darkkhaki= getColor("#BDB76B")
darkmagenta= getColor("#8B008B")
darkolivegreen= getColor("#556B2F")
darkorange= getColor("#FF8C00")
darkorchid= getColor("#9932CC")
darkred= getColor("#8B0000")
darksalmon= getColor("#E9967A")
darkseagreen= getColor("#8FBC8F")
darkslateblue= getColor("#483D8B")
darkslategray= getColor("#2F4F4F")
darkturquoise= getColor("#00CED1")
darkviolet= getColor("#9400D3")
deeppink= getColor("#FF1493")
deepskyblue= getColor("#00BFFF")
dimgray= getColor("#696969")
dodgerblue= getColor("#1E90FF")
firebrick= getColor("#B22222")
floralwhite= getColor("#FFFAF0")
forestgreen= getColor("#228B22")
fuchsia= getColor("#FF00FF")
gainsboro= getColor("#DCDCDC")
ghostwhite= getColor("#F8F8FF")
gold= getColor("#FFD700")
goldenrod= getColor("#DAA520")
greenyellow= getColor("#ADFF2F")
honeydew= getColor("#F0FFF0")
hotpink= getColor("#FF69B4")
indianred= getColor("#CD5C5C")
indigo= getColor("#4B0082")
ivory= getColor("#FFFFF0")
khaki= getColor("#F0E68C")
lavender= getColor("#E6E6FA")
lavenderblush= getColor("#FFF0F5")
lawngreen= getColor("#7CFC00")
lemonchiffon= getColor("#FFFACD")
lightcoral= getColor("#F08080")
lightcyan= getColor("#E0FFFF")
lightgray= getColor("#D3D3D3")
lightgreen= getColor("#90EE90")
lightpink= getColor("#FFB6C1")
lightsalmon= getColor("#FFA07A")
lightseagreen= getColor("#20B2AA")
lightskyblue= getColor("#87CEFA")
lightsteelblue= getColor("#B0C4DE")
lightyellow= getColor("#FFFFE0")
lime= getColor("#00FF00")
limegreen= getColor("#32CD32")
linen= getColor("#FAF0E6")
maroon= getColor("#800000")
mediumaquamarine= getColor("#66CDAA")
mediumblue= getColor("#0000CD")
mediumorchid= getColor("#BA55D3")
mediumpurple= getColor("#9370DB")
mediumseagreen= getColor("#3CB371")
mediumslateblue= getColor("#7B68EE")
mediumspringgreen= getColor("#00FA9A")
mediumturquoise= getColor("#48D1CC")
mediumvioletred= getColor("#C71585")
midnightblue= getColor("#191970")
mintcream= getColor("#F5FFFA")
mistyrose= getColor("#FFE4E1")
moccasin= getColor("#FFE4B5")
navajowhite= getColor("#FFDEAD")
navy= getColor("#000080")
oldlace= getColor("#FDF5E6")
olive= getColor("#808000")
olivedrab= getColor("#6B8E23")
orangered= getColor("#FF4500")
orchid= getColor("#DA70D6")
palegoldenrod= getColor("#EEE8AA")
palegreen= getColor("#98FB98")
paleturquoise= getColor("#AFEEEE")
palevioletred= getColor("#DB7093")
papayawhip= getColor("#FFEFD5")
peachpuff= getColor("#FFDAB9")
peru= getColor("#CD853F")
plum= getColor("#DDA0DD")
powderblue= getColor("#B0E0E6")
rebeccapurple= getColor("#663399")
rosybrown= getColor("#BC8F8F")
royalblue= getColor("#4169E1")
saddlebrown= getColor("#8B4513")
salmon= getColor("#FA8072")
sandybrown= getColor("#F4A460")
seagreen= getColor("#2E8B57")
seashell= getColor("#FFF5EE")
sienna= getColor("#A0522D")
silver= getColor("#C0C0C0")
skyblue= getColor("#87CEEB")
slateblue= getColor("#6A5ACD")
slategray= getColor("#708090")
snow= getColor("#FFFAFA")
blackboard= getColor("#393939")
springgreen= getColor("#00FF7F")
steelblue= getColor("#4682B4")
tan= getColor("#D2B48C")
thistle= getColor("#D8BFD8")
tomato= getColor("#FF6347")
turquoise= getColor("#40E0D0")
violet= getColor("#EE82EE")
wheat= getColor("#F5DEB3")
whitesmoke= getColor("#F5F5F5")
yellow= getColor("#FFFF00")
yellowgreen= getColor("#9ACD32")


__all__ = [


	'aliceblue',
	'antiquewhite',
	'aqua',
	'aquamarine',
	'azure',
	'beige',
	'bisque',
	'blanchedalmond',
	'blue',
	'blueviolet',
	'brown',
	'burlywood',
	'cadetblue',
	'chartreuse',
	'chocolate',
	'coral',
	'cornflowerblue',
	'cornsilk',
	'crimson',
	'cyan',
	'darkblue',
	'darkcyan',
	'darkgoldenrod',
	'darkgray',
	'darkgreen',
	'darkkhaki',
	'darkmagenta',
	'darkolivegreen',
	'darkorange',
	'darkorchid',
	'darkred',
	'darksalmon',
	'darkseagreen',
	'darkslateblue',
	'darkslategray',
	'darkturquoise',
	'darkviolet',
	'deeppink',
	'deepskyblue',
	'dimgray',
	'dodgerblue',
	'firebrick',
	'floralwhite',
	'forestgreen',
	'fuchsia',
	'gainsboro',
	'ghostwhite',
	'gold',
	'goldenrod',
	'greenyellow',
	'honeydew',
	'hotpink',
	'indianred',
	'indigo',
	'ivory',
	'khaki',
	'lavender',
	'lavenderblush',
	'lawngreen',
	'lemonchiffon',
	'lightcoral',
	'lightcyan',
	'lightgray',
	'lightgreen',
	'lightpink',
	'lightsalmon',
	'lightseagreen',
	'lightskyblue',
	'lightsteelblue',
	'lightyellow',
	'lime',
	'limegreen',
	'linen',
	'maroon',
	'mediumaquamarine',
	'mediumblue',
	'mediumorchid',
	'mediumpurple',
	'mediumseagreen',
	'mediumslateblue',
	'mediumspringgreen',
	'mediumturquoise',
	'mediumvioletred',
	'midnightblue',
	'mintcream',
	'mistyrose',
	'moccasin',
	'navajowhite',
	'navy',
	'oldlace',
	'olive',
	'olivedrab',
	'orangered',
	'orchid',
	'palegoldenrod',
	'palegreen',
	'paleturquoise',
	'palevioletred',
	'papayawhip',
	'peachpuff',
	'peru',
	'plum',
	'powderblue',
	'rebeccapurple',
	'rosybrown',
	'royalblue',
	'saddlebrown',
	'salmon',
	'sandybrown',
	'seagreen',
	'seashell',
	'sienna',
	'silver',
	'skyblue',
	'slateblue',
	'slategray',
	'snow',
	'blackboard',
	'springgreen',
	'steelblue',
	'tan',
	'thistle',
	'tomato',
	'turquoise',
	'violet',
	'wheat',
	'whitesmoke',
	'yellow',
	'yellowgreen',
	'red',
	'blue',
	'green',
	'orange',
	'pink',
	'magenta',
	'purple',
	'white',
	'black',
	'grey',
	'darkgrey',
	'teal',
	'lilla',
	'lightblue',

]
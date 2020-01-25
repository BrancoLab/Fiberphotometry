import sys
sys.path.append("./")

import os 
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from utils.maths.filtering import *
from utils.maths.stimuli_detection import *
from utils.colors import *
from analysis_utils import get_files_in_folder

# ? Define a bunch of colors
blueled = lightskyblue
violetled = violet


folder = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/fiberphot_data/200124_fiberp/240120_id_994382_freelymoving_twofibers_3'
files = get_files_in_folder(folder)

a = 1



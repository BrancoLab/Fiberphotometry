import sys
sys.path.append('./')

import os

from utils.file_io import listdir

def get_files_in_folder(folder):
    files = listdir(folder)

    behavcam = [f for f in files if 'cam1' in f][0]
    cacam = [f for f in files if 'cam0' in f][0]
    sensors = [f for f in files if 'sensors_data.csv' in f][0]

    return dict(behaviour=behavcam, calcium=cacam, sensors=sensors)
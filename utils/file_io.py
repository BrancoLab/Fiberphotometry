import sys
import os
import numpy as np
import csv
import pandas as pd
import yaml

def listdir(folderpath):
    return [os.path.join(folderpath,f) for f in os.listdir(folderpath)]

def get_subdirs(folderpath):
    return [ f.path for f in os.scandir(folderpath) if f.is_dir() ]

def check_create_folder(folderpath):
    # Check if a folder exists, otherwise creates it
    if not os.path.isdir(folderpath):
        os.mkdir(folderpath)

def check_folder_empty(folderpath):
    if not len(os.listdir(folderpath)):
        return True
    else:
        return False

def check_file_exists(filepath):
    # Check if a file with the given path exists already
    return os.path.isfile(filepath)


def create_csv_file(filepath, fieldnames):
    with open(filepath, "a", newline='') as f:
        logger = csv.DictWriter(f, fieldnames=fieldnames)
        logger.writeheader()

def append_csv_file(csv_file, row, fieldnames):
    with open(csv_file, "a", newline='') as f:
        logger = csv.DictWriter(f, fieldnames=fieldnames)
        logger.writerow(row)

def load_csv_file(csv_file):
    return pd.read_csv(csv_file)

def load_yaml(file):
        if not isinstance(file, str): raise ValueError('Invalid input argument')
        with open(file, 'r') as f:
                try:
                    loaded = yaml.full_load(f)
                except: loaded = yaml.load(f)
        return loaded

def save_yaml(path, obj, mode='w'):
    try:
        with open(path, mode) as f:
            yaml.dump(obj, f)
    except: return False
    else: return True

def save_df(df, filepath):
        df.to_pickle(filepath)

def load_df(filepath):
        return pd.read_pickle(filepath)
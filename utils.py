# Random utility functions for OpenKE grid search
import os, shutil, json

def create_directory(dir):
    ''' 
    Creates directory - overwrites existing one if it exists and creates new one otherwise 
    ''' 
    
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_json(obj, fpath):
    with open(fpath, 'w') as out_file:
        json.dump(obj, out_file)
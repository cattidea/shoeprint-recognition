import os

from utils.config import Config
from utils.imager import image2array

CONFIG = Config()
SIMPLE_DIR = CONFIG['simple_dir']
DETERMINE_FILE = CONFIG["determine_file"]
SHOEPRINT_DIR = CONFIG["shoeprint_dir"]

def get_simple_index():
    simple_index = {}
    for t in os.listdir(SIMPLE_DIR):
        type_dir = os.path.join(SIMPLE_DIR, t)
        img_path = os.path.join(type_dir, os.listdir(type_dir)[0])
        simple_index[t] = img_path
    return simple_index

def get_determine():
    determine = {}
    with open(DETERMINE_FILE, 'r') as f:
        for line in f:
            line_items = line.split('\t')
            for i in range(len(line_items)):
                line_items[i] = line_items[i].strip()
            determine[line_items[0]] = line_items[1: ]
    return determine

def get_shoeprints(determine, simple_index):
    shoeprints = []
    types = os.listdir(SHOEPRINT_DIR)
    for t in types:
        type_dir = os.path.join(SHOEPRINT_DIR, t)
        for filename in os.listdir(type_dir):
            file_path = os.path.join(type_dir, filename)
            if filename in determine:
                determine_list = determine[filename]
                assert t in determine_list
                for simple_type in determine_list:
                    if simple_type == t:
                        continue
                    shoeprints.append([file_path, simple_index[t], simple_index[simple_type]])
    return shoeprints

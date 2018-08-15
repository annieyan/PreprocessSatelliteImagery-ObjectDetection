'''
Get the statistics of training data and testing data
1) # of bounding boxes in train / test /total
2) # of 2048 x 2048 chips in train/ test /total
3) # of 512 x 512 chips in train / test / total 
4) # of examples each class
'''

import argparse
import os
import aug_util as aug
import wv_util as wv
import matplotlib.pyplot as plt
import numpy as np
import csv
#import matplotlib, copy, skimage, os, tifffile
from skimage import io, morphology, draw
import gdal
from PIL import Image
import random
import json
from tqdm import tqdm
import io
import glob
import shutil
import os
import geopandas as gpd
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import pandas as pd

# get bbox count for geojson with ONLY DAMAGED buildings
# get unique chips count
def get_bbox_count(fname):
    with open(fname) as f:
        data = json.load(f)

    coords = np.zeros((len(data['features']),4))
    chips = np.zeros((len(data['features'])),dtype="object")
    classes = np.zeros((len(data['features'])))
    # debug
    uids = np.zeros((len(data['features'])))

    for i in tqdm(range(len(data['features']))):
        if data['features'][i]['properties']['bb'] != []:
            try: 
                b_id = data['features'][i]['properties']['IMAGE_ID']
#                 if b_id == '20170831_105001000B95E100_3020021_jpeg_compressed_06_01.tif':
#                     print('found chip!')
                bbox = data['features'][i]['properties']['bb'][1:-1].split(",")
                val = np.array([int(num) for num in data['features'][i]['properties']['bb'][1:-1].split(",")])
                
                ymin = val[3]
                ymax = val[1]
                val[1] =  ymin
                val[3] = ymax
                chips[i] = b_id
                classes[i] = data['features'][i]['properties']['TYPE_ID']
                # debug
                uids[i] = int(data['features'][i]['properties']['bb_uid'])
            except:
#                 print('i:', i)
#                 print(data['features'][i]['properties']['bb'])
                  pass
            if val.shape[0] != 4:
                print("Issues at %d!" % i)
            else:
                coords[i] = val
        else:
            chips[i] = 'None'
    # debug
    # added offsets to each coordinates
    # need to check the validity of bbox maybe
    # debug
    # mute the shifting of bbox for now
    # because no need of adjusting bbox in statistics
    #coords = np.add(coords, add_np)
    # get the count of unique chips
    chip_unique = len(np.unique(chips))
    print('The total number of bboxes for training + test: ', len(data['features']))
    print('The total number of 2048 chips for training + test: ', chip_unique)

    
    return coords, chips, classes, uids


# for tomnod + MS data, 2 classes
def get_bbox_count_multiclass(fname):
    """
    Gets label data from a geojson label file
    Args:
        fname: file path to an xView geojson label file
    Output:
        Returns three arrays: coords, chips, and classes corresponding to the
            coordinates, file-names, and classes for each ground truth.
    """
      # debug
    x_off = 15
    y_off = 15
    right_shift = 5 # how much shift to the right 
    add_np = np.array([-x_off + right_shift, -y_off, x_off + right_shift, y_off])  # shift to the rihgt
    with open(fname) as f:
        data = json.load(f)

    coords = np.zeros((len(data['features']),4))
    chips = np.zeros((len(data['features'])),dtype="object")
    classes = np.zeros((len(data['features'])))
    # debug
    uids = np.zeros((len(data['features'])))

    for i in tqdm(range(len(data['features']))):
        if data['features'][i]['properties']['bb'] != []:
            try: 
                b_id = data['features'][i]['properties']['Joined lay']
                bbox = data['features'][i]['properties']['bb'][1:-1].split(",")
                val = np.array([int(num) for num in data['features'][i]['properties']['bb'][1:-1].split(",")])
                
                chips[i] = b_id
                classes[i] = data['features'][i]['properties']['type']
                # debug
                uids[i] = int(data['features'][i]['properties']['uniqueid'])
            except:
                  pass
            if val.shape[0] != 4:
                print("Issues at %d!" % i)
            else:
                coords[i] = val
        else:
            chips[i] = 'None'
    # debug
    # added offsets to each coordinates
    # need to check the validity of bbox maybe
    chip_unique = len(np.unique(chips))
    print('The total number of bboxes for training + test: ', len(data['features']))
    print('The total number of bboxes for damaged buildings  training + test: ', classes[classes == 1].shape[0])
    print('The total number of bboxes for non-damaged buildings  training + test: ', classes[classes == 2].shape[0])

    print('The total number of 2048 chips for training + test: ', chip_unique)
    return coords, chips, classes, uids






def parse_args():
    """Parse command line arguments passed to script invocation."""
    parser = argparse.ArgumentParser(
        description='Get statistics for training data and test data from geojson and tif images.')

    parser.add_argument('src_geojson', help='source geojson')

    return parser.parse_args()


def main():
    args = parse_args()
    geojson_path = args.src_geojson
    #geojson_path = '../just_buildings_w_uid_second_round.geojson'
    get_bbox_count_multiclass(geojson_path)



if __name__ == "__main__":
    main()





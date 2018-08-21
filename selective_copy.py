'''
Copyright (C) 2018 <eScience Institue at University of Washington>
Licensed under CC BY-NC-ND 4.0 License [see LICENSE-CC BY-NC-ND 4.0.markdown for details] 
Written by An Yan

'''

# select image chips that are with digital globe labels



import wv_util as wv
import matplotlib.pyplot as plt
import numpy as np
import csv
#%matplotlib inline
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



def get_unique_images(fname):
    with open(fname) as f:
        data = json.load(f)

    coords = np.zeros((len(data['features']),4))
    chips = np.zeros((len(data['features'])),dtype="object")
    classes = np.zeros((len(data['features'])))

    for i in tqdm(range(len(data['features']))):
        if data['features'][i]['properties']['bb'] != []:
            try:
                b_id = data['features'][i]['properties']['Joined lay']
                if b_id == '20170902_10400100324DAE00_3210111_jpeg_compressed_09_05.tif':
                    print('found chip!')
                bbox = data['features'][i]['properties']['bb'][1:-1].split(",")

                val = np.array([int(num) for num in data['features'][i]['properties']['bb'][1:-1].split(",")])

                ymin = val[3]
                ymax = val[1]
                val[1] =  ymin
                val[3] = ymax
                #print(val)
                chips[i] = str(b_id)

                classes[i] = data['features'][i]['properties']['type']
            except:
                print('i:', i)
                print(data['features'][i]['properties']['Joined lay'])
                  #pass
            if val.shape[0] != 4:
                print("Issues at %d!" % i)
            else:
                coords[i] = val
        else:
            chips[i] = 'None'
            print('warning: chip is none')
    unique_image_set = set(chips.tolist())
    return unique_image_set




def main():

    geojson_file = '../harvey_test_second_noblack_ms_noclean.geojson'
    unique_image_set = get_unique_images(geojson_file)
    print('number of chips is :', len(unique_image_set))

    path = '/home/ubuntu/anyan/harvey_data/harvey_test_second_noblack/'
    save_path =  '/home/ubuntu/anyan/harvey_data/harvey_test_bigtiff_v3/'


    files = [os.path.join(path, f) for f in os.listdir(path)]

    i = 0
    #parent_folder = os.path.abspath(abs_dirname + "/../")
    #seperate_subdir = None
    #subdir_name = os.path.join(parent_folder, 'train_small')
    #seperate_subdir = subdir_name
    #os.mkdir(subdir_name)

    for f in files:
        # create new subdir if necessary
        
            #subdir_name = os.path.join(abs_dirname, '{0:03d}'.format(i / N + 1))
           # os.mkdir(subdir_name)
           # seperate_subdir = subdir_name

        # copy file to current dir
        f_base = os.path.basename(f)
        if f_base in unique_image_set:
            print('filename: ', f_base)
            shutil.copy(f, os.path.join(save_path, f_base))
            i += 1
            print('copied: ', i)



if __name__ == '__main__':
    main()

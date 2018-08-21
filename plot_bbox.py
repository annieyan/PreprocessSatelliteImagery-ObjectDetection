'''
Copyright (C) 2018 <eScience Institue at University of Washington>
Licensed under CC BY-NC-ND 4.0 License [see LICENSE-CC BY-NC-ND 4.0.markdown for details] 
Written by An Yan

'''


'''
Take 2048 * 2048 chips, overlay bounding boxes with uids on them,
and output png of the same size
'''

import aug_util as aug
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



# This is for single class Tomnod + Oak Ridge building foot print data (no offsets)
def get_labels(fname):
    """
    Gets label data from a geojson label file
    Args:
        fname: file path to an xView geojson label file
    Output:
        Returns three arrays: coords, chips, and classes corresponding to the
            coordinates, file-names, and classes for each ground truth.
    """
    with open(fname) as f:
        data = json.load(f)

    coords = np.zeros((len(data['features']),4))
    chips = np.zeros((len(data['features'])),dtype="object")
    classes = np.zeros((len(data['features'])))

    for i in tqdm(range(len(data['features']))):
        if data['features'][i]['properties']['bb'] != []:
            try: 
                b_id = data['features'][i]['properties']['IMAGE_ID']
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
                
                classes[i] = data['features'][i]['properties']['TYPE_ID']
            except:
                print('i:', i)
                print(data['features'][i]['properties']['IMAGE_ID'])
                  #pass
            if val.shape[0] != 4:
                print("Issues at %d!" % i)
            else:
                coords[i] = val
        else:
            chips[i] = 'None'
            print('warning: chip is none')

    return coords, chips, classes


# This is for 2 classes data: damaged + non-damaged buildings
# Suitable for Tomnod + MS building footprints (no offsets)
def get_labels_w_uid_nondamaged(fname):
    """
    Gets label data from a geojson label file
    Args:
        fname: file path to an xView geojson label file
    Output:
        Returns three arrays: coords, chips, and classes corresponding to the
            coordinates, file-names, and classes for each ground truth.
    """
      # debug
#     x_off = 15
#     y_off = 15
#     right_shift = 5  # how much shift to the right 
#     add_np = np.array([-x_off + right_shift, -y_off, x_off + right_shift, y_off])  # shift to the rihgt
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
#                 if b_id == '20170831_105001000B95E100_3020021_jpeg_compressed_06_01.tif':
#                     print('found chip!')
                bbox = data['features'][i]['properties']['bb'][1:-1].split(",")
                val = np.array([int(num) for num in data['features'][i]['properties']['bb'][1:-1].split(",")])
                
#                 ymin = val[3]
#                 ymax = val[1]
#                 val[1] =  ymin
#                 val[3] = ymax
                chips[i] = b_id
                classes[i] = data['features'][i]['properties']['type']
                # debug
                uids[i] = int(data['features'][i]['properties']['uniqueid'])
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
    #coords = np.add(coords, add_np)
    
    return coords, chips, classes, uids





def draw_bbox_on_tiff(chip_path, coords, chips, classes,uids, save_path):
    #Load an image
    #path = '/home/ubuntu/anyan/harvey_data/converted_sample_tiff/'
    
    
    # big tiff name: chip name
    # init to {big_tiff_name : []}
    #big_tiff_dict = dict((k, []) for k in big_tiff_set)
    #big_tiff_dict = dict()

    fnames = glob.glob(chip_path + "*.tif")
    i = 0
    for f in fnames:
        
        chip_name = f.split('/')[-1].strip()
        chip_big_tiff_id_list = chip_name.split('_')[1:3]
        chip_big_tiff_id = '_'.join(chip_big_tiff_id_list)
        #print(chip_big_tiff_id)
        '''
        if chip_big_tiff_id not in set(big_tiff_dict.keys()):
            big_tiff_dict[chip_big_tiff_id] = list()
            big_tiff_dict[chip_big_tiff_id].append(chip_name)
        else:
            big_tiff_dict[chip_big_tiff_id].append(chip_name)        
        
        if len(big_tiff_dict[chip_big_tiff_id]) > 5:
            continue
        '''
            # debug
        print(chip_big_tiff_id)
            #big_tiff_dict[chip_big_tiff_id].append(chip_name)
        arr = wv.get_image(f)
#             print(arr.shape)
    #         plt.figure(figsize=(10,10))
    #         plt.axis('off')
    #         plt.imshow(arr)
        coords_chip = coords[chips==chip_name]
        #print(chip_name)
            #print(coords_chip.shape)
        if coords_chip.shape[0] == 0:
            print('no bounding boxes in this image')
            print(chip_name)
            continue
        classes_chip = classes[chips==chip_name].astype(np.int64)
    #         #We can chip the image into 500x500 chips
    #         c_img, c_box, c_cls = wv.chip_image(img = arr, coords= coords, classes=classes, shape=(500,500))
    #         print("Num Chips: %d" % c_img.shape[0])
        uids_chip = uids[chips == chip_name].astype(np.int64)
        labelled = aug.draw_bboxes_withindex_multiclass(arr,coords_chip,classes_chip, uids_chip)
        print(chip_name)
#             plt.figure(figsize=(15,15))
#             plt.axis('off')
#             plt.imshow(labelled)
        subdir_name = save_path + chip_big_tiff_id
        if os.path.isdir(subdir_name):
            save_name = subdir_name +'/' + chip_name + '.png'
            print('saving image: ', save_name)
            labelled.save(save_name)
        else:
            os.mkdir(subdir_name)
            save_name = subdir_name +'/' + chip_name + '.png'
            print('saving image: ',save_name)
            labelled.save(save_name)
           
            
        #else:
            #continue
        
        #debug
        #print('len of big_tiff_dict: ', len(big_tiff_dict.keys()))
   
        #chip_name = '20170831_105001000B95E100_3020021_jpeg_compressed_06_01.tif'
        # chip_name = '20170831_105001000B95E100_3020021_jpeg_compressed_04_04.tif'
        #chip_name = '20170831_105001000B95E100_3020021_jpeg_compressed_05_02.tif'
        #chip_name = '20170831_105001000B95E100_3020021_jpeg_compressed_05_04.tif'
        #chip_name = '20170831_105001000B95E100_3020021_jpeg_compressed_06_02.tif'
        #chip_fullname = path + chip_name
        #print(chip_fullname)
       
   
            #labelled.save("test.png")







def main():


    #geojson_file = '../bounding_box_referenced_2.geojson'
    #geojson_file = '../harvey_test_second.geojson'
    geojson_file = '../bboxes_tomnod_2class_v1.geojson'
    coords, chips, classes, uids = wv.get_labels_w_uid_nondamaged(geojson_file)
    print('number of chips is :', chips.shape)
    test_tif = '20170902_10400100324DAE00_3210111_jpeg_compressed_09_05.tif'
    if test_tif in chips.tolist():
        print('test tif exists!!!!!')

    #print('chips, ', chips.tolist())
    #path = '/home/ubuntu/anyan/harvey_data/harvey_test_second/'
    #save_path =  '/home/ubuntu/anyan/harvey_data/inspect_black_in_test/'
    path = '../harvey_vis_result_toydata/'
    save_path = '../harvey_vis_result_toydata_bboxes/'

    #aug.draw_bboxes_withindex(arr,coords_chip, uids_chip)
    draw_bbox_on_tiff(path, coords, chips, classes,uids, save_path)

if __name__ == '__main__':
    main()

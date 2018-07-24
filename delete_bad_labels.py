'''
This is to delete bad bboxes from geojson that manually identified
1. Use the original geojson and training/test chips, run process_wv.py to produce
   512 x 512 small chips with bboxes in a folder called 'harvey_img_inspect_train (test)'
2. manually inspect every small chips to identify bad bboxes. move bad chips to a folder
3. get a list of 2048 x 2048 chip names that contain bad examples, plot bboxes upon them 
   along with bbox uid
4. manually create a csv, record uids to be deleted.
5. remove uids from geojson and create a new geojson
'''


'''
identify_bad_labels.py does

1. from a folder of bad labels in small tifs, find out big tiff (2048) ids 
   plot bbox over these big tiff with uid


delete_bad_labels.py does: 
2. after manual inspection. Take a list of bad uid, delete them from geojson and form
   a new geojson
'''

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

# modified to buffer the bounding boxes by 15 pixels
# return uids of bboxes as well 
def get_labels_w_uid(fname):
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
    right_shift = 5  # how much shift to the right 
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
    coords = np.add(coords, add_np)
    
    return coords, chips, classes, uids




# draw bboxes with bbox uid

def draw_bboxes_withindex(img,boxes, uids):
    """
    A helper function to draw bounding box rectangles on images
    Args:
        img: image to be drawn on in array format
        boxes: An (N,4) array of bounding boxes
    Output:
        Image with drawn bounding boxes
    """
    source = Image.fromarray(img)
    draw = ImageDraw.Draw(source)
    w2,h2 = (img.shape[0],img.shape[1])
    
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 40)
    #font = ImageFont.truetype('arial.ttf', 24)


    idx = 0

    for b in boxes:
        xmin,ymin,xmax,ymax = b
        
        for j in range(3):
            draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="red")
        draw.text((xmin+20, ymin+70), str(uids[idx]), font = font)
        idx +=1
    return source




# delete bboxes based on bbox uid
# return new geojson
# fname = '../just_buildings_w_uid.geojson'
def delete_bbox_from_geojson(old_geojson, rows_to_delete):
    gfN = gpd.read_file(old_geojson)
    index_list = []
    df_len = len(gfN)
    

    for i in range(0, df_len):
        print('idx', i)
        series_tmp = gfN.loc[i]
        if series_tmp['bb_uid'] in set(rows_to_delete):
            continue
        index_list.append(i)
    geometries = [xy for xy in list(gfN.iloc[index_list]['geometry'])]
    crs = {'init': 'epsg:4326'}
    gf = gpd.GeoDataFrame(gfN.iloc[index_list], crs=crs, geometry=geometries)

# geometries = [shapely.geometry.Point(xy) for xy in zip(df.lng, df.lat)]
# gf = gpd.GeoDataFrame(gfN.iloc[0],)
    parent_folder = os.path.abspath(old_geojson + "/../")

    # get training or test dir name
    f_base = os.path.basename(old_geojson)
    
    save_name = ''.join(f_base.split('.')[0:-1]) +'_cleaned.geojson'
    print(save_name)
    gf.to_file(parent_folder+'/'+ save_name, driver='GeoJSON')
     
 
# take a list of big tifs containing bad labels, draw bboxes 
# with uid, and save as png
# args: 
#  big_chip_list :  list of individual big tif file names,
#  chip_path : path to all big chip files

def draw_bbox_on_tiff(bad_big_tif_list, chip_path, coords, chips, classes, uids, save_path):
    

    i = 0
    # f is the filename without path, chip_name is the filename + path
    for f in bad_big_tif_list:
        if len(f) < 5:
            print('invalid file name: ', f)
            continue
        chip_name = os.path.join(chip_path, f)
        print(chip_name)
        
        arr = wv.get_image(chip_name)
        coords_chip = coords[chips==f]
        if coords_chip.shape[0] == 0:
            print('no bounding boxes for this image')
            continue
        classes_chip = classes[chips==f].astype(np.int64)
        uids_chip = uids[chips == f].astype(np.int64)
        labelled = draw_bboxes_withindex(arr,coords_chip[classes_chip ==1], uids_chip)
        print(chip_name)
#             plt.figure(figsize=(15,15))
#             plt.axis('off')
#             plt.imshow(labelled)
        subdir_name = save_path
        if os.path.isdir(subdir_name):
            save_name = subdir_name +'/' + f + '.png'
            print(save_name)
            labelled.save(save_name)
        else:
            os.mkdir(subdir_name)
            save_name = subdir_name +'/' + f + '.png'
            print(save_name)
            labelled.save(save_name)
           

#  read all files in a folder which contains small tifs that have
#  bad labels. Get all big tiff names in a list
#  args: small_tif_dir: absolute path to a dir containing bad small tifs
#  return: a list (set) of big tiff names / paths
#  i.e.  img_20170829_1040010032211E00_2110222_jpeg_compressed_09_04.tif_8.png
def parse_tif_names(small_tif_dir):
    #files = [os.path.join(small_tif_dir, f) for f in os.listdir(small_tif_dir)]
    fnames = [f for f in os.listdir(small_tif_dir)]
    bad_big_tif_list = set()
    for fname in fnames:
        big_tif_name = fname.split('.')[0]
        big_tif_name = '_'.join(big_tif_name.split('_')[1:])
        big_tif_name = big_tif_name + '.tif'
        bad_big_tif_list.add(big_tif_name)
    return bad_big_tif_list


# args: a csv file containing small image names which contains bad labels
# read and parse the big tif names
def parse_tif_names_from_csv(path_to_csv):
    #files = [os.path.join(small_tif_dir, f) for f in os.listdir(small_tif_dir)]
    #fnames = [f for f in os.listdir(small_tif_dir)]
    bad_small_tifs = pd.read_csv(path_to_csv)
    bad_small_list = set(bad_small_tifs['bad_label'].tolist())


    bad_big_tif_list = set()
    for fname in bad_small_list:
        big_tif_name = fname.split('.')[0]
        big_tif_name = '_'.join(big_tif_name.split('_')[1:])
        big_tif_name = big_tif_name + '.tif'
        bad_big_tif_list.add(big_tif_name)
    return bad_big_tif_list




def main():
    # RUN THIS CHUNK TO INSPECT BIG TIFFs WITH BBOXES
    # read tif
    #small_tif_dir = '/home/ubuntu/anyan/harvey_data/bad_labels_small'
    
    #geojson_file = '../just_buildings_w_uid.geojson'
    geojson_file = '../added_non_damaged.geojson'


    #coords, chips, classes, uids  = get_labels_w_uid(geojson_file)
    #bad_big_tif_list = parse_tif_names(small_tif_dir)
    #print('len of bad big tifs: ', len(bad_big_tif_list))
    #print(bad_big_tif_list)
    # draw bbox with uid on big tiffs, save as png
    #save_path = '/home/ubuntu/anyan/harvey_data/bad_labels_big'
    #chip_path = '/home/ubuntu/anyan/harvey_data/filtered_converted_image_buildings'
    #draw_bbox_on_tiff(bad_big_tif_list, chip_path, coords, chips, classes, uids, save_path)
    #print('len of bad big tifs: ', len(bad_big_tif_list))

    # RUN THIS TO DELETE UIDS FROM GEOJSON
    # read bad labels from file
    bad_label_path = '../bad_labels.csv'
    bad_label_df = pd.read_csv(bad_label_path)
    bad_label_list = set(bad_label_df['bad_label'].tolist())
    
   

    # delete from geoson
    delete_bbox_from_geojson(geojson_file, bad_label_list)
    print('len of bad lables: ', len(bad_label_list))
    


if __name__ == "__main__":
    main()

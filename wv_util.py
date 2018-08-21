"""
Copyright 2018 Defense Innovation Unit Experimental
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications copyright (C) 2018 <eScience Institue at University of Washington>
Licensed under CC BY-NC-ND 4.0 License [see LICENSE-CC BY-NC-ND 4.0.markdown for details] 
Written by An Yan
"""

from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import aug_util as aug
import random


"""
xView processing helper functions for use in data processing.
"""

def scale(x,range1=(0,0),range2=(0,0)):
    """
    Linear scaling for a value x
    """
    return range2[0]*(1 - (x-range1[0]) / (range1[1]-range1[0])) + range2[1]*((x-range1[0]) / (range1[1]-range1[0]))


def get_image(fname):    
    """
    Get an image from a filepath in ndarray format
    """
    return np.array(Image.open(fname))

'''
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
        if data['features'][i]['properties']['bounds_imcoords'] != []:
            b_id = data['features'][i]['properties']['image_id']
            val = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
            chips[i] = b_id
            classes[i] = data['features'][i]['properties']['type_id']
            if val.shape[0] != 4:
                print("Issues at %d!" % i)
            else:
                coords[i] = val
        else:
            chips[i] = 'None'

    return coords, chips, classes
'''



# get labels for noaa data#
def get_labels_noaa_w_uids(fname):
    """
    Gets label data from a geojson label file
    Args:
        fname: file path to a NOAA data geojson label file
    Output:
        Returns three arrays: coords, chips, and classes corresponding to the
            coordinates, file-names, and classes for each ground truth.
    """
    x_off = 10
    y_off = 10
    add_np = np.array([-x_off, -y_off, x_off, y_off])
    with open(fname) as f:
        data = json.load(f)

    coords = np.zeros((len(data['features']),4))
    chips = np.zeros((len(data['features'])),dtype="object")
    classes = np.zeros((len(data['features'])))
    uids = np.zeros((len(data['features'])))

    for i in tqdm(range(len(data['features']))):
        if data['features'][i]['properties']['bb'] != []:
            try: 
                full_imgid = data['features'][i]['properties']['image']
                b_id = full_imgid.split('/')[-1]
                bbox = data['features'][i]['properties']['bb'][1:-1].split(",")
                val = np.array([int(num) for num in data['features'][i]['properties']['bb'][1:-1].split(",")])
                uids[i] = data['features'][i]['properties']['id']
                chips[i] = b_id
                classes[i] = data['features'][i]['properties']['type_id']
            except:
                  pass
            if val.shape[0] != 4:
                print("Issues at %d!" % i)
            else:
                coords[i] = val
        else:
            chips[i] = 'None'
    coords = np.add(coords, add_np)
    
    return coords, chips, classes, uids





# get labels for tomnod data
# modified to buffer the bounding boxes by 15 pixels
def get_labels(fname):
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

    for i in tqdm(range(len(data['features']))):
        if data['features'][i]['properties']['bb'] != []:
            try: 
                b_id = data['features'][i]['properties']['IMAGE_ID']
                bbox = data['features'][i]['properties']['bb'][1:-1].split(",")
                val = np.array([int(num) for num in data['features'][i]['properties']['bb'][1:-1].split(",")])
                
                ymin = val[3]
                ymax = val[1]
                val[1] =  ymin
                val[3] = ymax
                chips[i] = b_id
                classes[i] = data['features'][i]['properties']['TYPE_ID']
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
    coords = np.add(coords, add_np)
    
    return coords, chips, classes





# debug
# this is for Tomnod + Oak Ridge building footprint data
# modified to buffer the bounding boxes by 15 pixels, and shift to the right
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







# debug
# This is for Tomnox + Microsoft building footprint data 
# this is for geojson with 2 classes: damaged and non-damaged
# TODO: add offset
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
    coords = np.add(coords, add_np)
    
    return coords, chips, classes, uids





def boxes_from_coords(coords):
    """
    Processes a coordinate array from a geojson into (xmin,ymin,xmax,ymax) format

    Args:
        coords: an array of bounding box coordinates

    Output:
        Returns an array of shape (N,4) with coordinates in proper format
    """
    nc = np.zeros((coords.shape[0],4))
    for ind in range(coords.shape[0]):
        x1,x2 = coords[ind,:,0].min(),coords[ind,:,0].max()
        y1,y2 = coords[ind,:,1].min(),coords[ind,:,1].max()
        nc[ind] = [x1,y1,x2,y2]
    return nc




# given a 2048 tif and its labels, chip it to small images that centered with class 1
# bounding boxes and randomly select N out of all satisfying bboxes.
# Add random offsets to avoid placing bboxes at the center all the 
# other, otherwise models will overfit to the bbox in the center
# prob: the probability of selecting the number of chips to produce 
# for example, if there are 10 chips centered around class1
# The output will be 10 * (1/prob) images (if valid)
# prob should be a random int between 5 ~ 10, meaning 10% ~ 20% change of augmenting
def random_crop_from_center(img,coords_chip,classes_chip,  prob, resolution = (200,200)):
    w = img.shape[0]
    h = img.shape[1]
    crop_w, crop_h = resolution
    threshold = 20  # threshold of # of pixels to discard bbox
    boxes = np.array(coords_chip)
    print('number of bboxes: ', boxes.shape[0])
    images = np.zeros((coords_chip.shape[0],crop_w,crop_h,3))
    total_boxes = {}
    total_classes = {}
    k = 0
    # number of class 1 chips
    num_class1 = classes_chip[classes_chip ==1].shape[0]
    #num_aug = int(num_class1 * prob)

    for i in range(boxes.shape[0]):
        if classes_chip[i] == 2:
            continue
        p = np.random.randint(0,prob)
        if p > 2:
            continue
        xmin, ymin, xmax, ymax = boxes[i]
#         bbox_x_center = (xmin + xmax)/2
#         bbox_y_center = (ymin + ymax) /2
        bbox_y_center = (xmin + xmax)/2
        bbox_x_center = (ymin + ymax) /2

        # generate random offsets for x and y
        offset_x = random.randint(-40, 40)
        offset_y = random.randint(-40, 40)
        # force the crop to be square and contain the chosen bbox
        if bbox_x_center + offset_x < 1/2 * crop_w:
             # start from leftmost
            startx = 0
            endx = int(startx + crop_w)
            # should consider the case: if endx < xmax
        elif bbox_x_center + offset_x > w - 1/2 * crop_w:
            endx = w
            startx = int(w - crop_w)
        else:
            endx = int(bbox_x_center + + offset_x +  1/2 * crop_w)
            #startx = int(w - crop_w)
            startx = int(bbox_x_center + offset_x - 1/2 * crop_w)
        if bbox_y_center + offset_y < 1/2 * crop_h:
            starty = 0
            endy = int(starty + crop_h)
        elif bbox_y_center + offset_y > int(h - 1/2 *crop_h):
            endy = h
            starty = int(endy - crop_h)
        else:
            endy = int(bbox_y_center ++ offset_y+ 1/2 * crop_h)
            starty = int(bbox_y_center + offset_y - 1/2 * crop_h)
        newimg = img[startx: endx, starty: endy]
        newboxes = []
        newclasses = []

        #boxes = np.array(coords_chip)  # change to np array, otherwise, boxes[:,0] cannot access list
        x = np.logical_or( np.logical_and( (boxes[:,0]<endy),  (boxes[:,0]>starty)),
                                   np.logical_and((boxes[:,2]<endy),  (boxes[:,2]>starty)))
        out = boxes[x]
        y = np.logical_or( np.logical_and(  (out[:,1]<endx),  (out[:,1]>startx)),
                                   np.logical_and((out[:,3]<endx),  (out[:,3]>startx)))
        outn = out[y]
        out = np.transpose(np.vstack((np.clip(outn[:,0]-starty,0,crop_w),
                                              np.clip(outn[:,1]-startx,0, crop_h),
                                              np.clip(outn[:,2]-starty,0,crop_w),
                                              np.clip(outn[:,3]-startx,0, crop_h))))
        box_classes = classes_chip[x][y]
        # remove bboxes that only have less than 20 pixels in w/h left in the image
        # only loop through ones that have 0 or wn/hn in the 4 coordinates
        rows_to_delete = list()
        for m in range(out.shape[0]):
            if(np.any([out[m] == 0]) or np.any([out[m] == crop_w]) or np.any([out[m] == crop_h])):
             # see whether the width of bbox is less than 10 pixels?
                bbox_w = out[m][2] - out[m][0]
                bbox_h = out[m][3] - out[m][1]
                if bbox_w < threshold or bbox_h < threshold:
                    rows_to_delete.append(m)

        # discard this bbox

        out = np.delete(out, rows_to_delete, axis=0)
        box_classes = np.delete(box_classes, rows_to_delete, axis=0)

        if out.shape[0] != 0:
            newboxes = out
            newclasses = box_classes
        else:
            newboxes= np.array([[0,0,0,0]])
            newclasses = np.array([0])
            # check whether there are any bboxes on the image, if not, discard
        newimg, new_bboxes, new_classes = aug.check_bbox_validity(newimg, newboxes, newclasses)
        if len(new_bboxes) != 0:

            images[k] = newimg
            total_boxes[k] = new_bboxes
            total_classes[k] = new_classes

            print("processing round: ", k)
            k = k+1

    # only retain k images
    final_aug_num = len(total_boxes)
    print('final number: ',final_aug_num )
    images = images[0:final_aug_num]
    return images.astype(np.uint8),total_boxes,total_classes





# given a 2048 tif and its labels, chip it to small images that centered with each
# bounding boxes. Add random offsets to avoid placing bboxes at the center all the 
# other, otherwise models will overfit to the bbox in the center
def crop_from_center(img,coords_chip,classes_chip, uids_chip, resolution = (200,200)):
    w = img.shape[0]
    h = img.shape[1]
    crop_w, crop_h = resolution
    threshold = 20  # threshold of # of pixels to discard bbox
    boxes = np.array(coords_chip) 
    print('number of bboxes: ', boxes.shape[0])
    images = np.zeros((coords_chip.shape[0],crop_w,crop_h,3))
    total_boxes = {}
    total_classes = {}
    k = 0
    for i in range(boxes.shape[0]):
        xmin, ymin, xmax, ymax = boxes[i]
#         bbox_x_center = (xmin + xmax)/2
#         bbox_y_center = (ymin + ymax) /2
        bbox_y_center = (xmin + xmax)/2
        bbox_x_center = (ymin + ymax) /2

        # generate random offsets for x and y
        offset_x = random.randint(-40, 40)
        offset_y = random.randint(-40, 40)
        # force the crop to be square and contain the chosen bbox
        if bbox_x_center + offset_x < 1/2 * crop_w:
             # start from leftmost
            startx = 0
            endx = int(startx + crop_w)
            # should consider the case: if endx < xmax
        elif bbox_x_center + offset_x > w - 1/2 * crop_w:
            endx = w
            startx = int(w - crop_w)
           
        else:
            endx = int(bbox_x_center + + offset_x +  1/2 * crop_w)
            #startx = int(w - crop_w)
            startx = int(bbox_x_center + offset_x - 1/2 * crop_w)
        if bbox_y_center + offset_y < 1/2 * crop_h:
            starty = 0
            endy = int(starty + crop_h)
        elif bbox_y_center + offset_y > int(h - 1/2 *crop_h):
            endy = h
            starty = int(endy - crop_h)
        else:
            endy = int(bbox_y_center ++ offset_y+ 1/2 * crop_h)
            starty = int(bbox_y_center + offset_y - 1/2 * crop_h)
        newimg = img[startx: endx, starty: endy]
        newboxes = []
        newclasses = []
       
        #boxes = np.array(coords_chip)  # change to np array, otherwise, boxes[:,0] cannot access list
        x = np.logical_or( np.logical_and( (boxes[:,0]<endy),  (boxes[:,0]>starty)),
                                   np.logical_and((boxes[:,2]<endy),  (boxes[:,2]>starty)))
        out = boxes[x]
        y = np.logical_or( np.logical_and(  (out[:,1]<endx),  (out[:,1]>startx)),
                                   np.logical_and((out[:,3]<endx),  (out[:,3]>startx)))
        outn = out[y]
        out = np.transpose(np.vstack((np.clip(outn[:,0]-starty,0,crop_w),
                                              np.clip(outn[:,1]-startx,0, crop_h),
                                              np.clip(outn[:,2]-starty,0,crop_w),
                                              np.clip(outn[:,3]-startx,0, crop_h))))
        box_classes = classes_chip[x][y]
        # remove bboxes that only have less than 20 pixels in w/h left in the image
        # only loop through ones that have 0 or wn/hn in the 4 coordinates
        rows_to_delete = list()
        for m in range(out.shape[0]):
            if(np.any([out[m] == 0]) or np.any([out[m] == crop_w]) or np.any([out[m] == crop_h])):
             # see whether the width of bbox is less than 10 pixels?
                bbox_w = out[m][2] - out[m][0]
                bbox_h = out[m][3] - out[m][1]
                if bbox_w < threshold or bbox_h < threshold:
                    rows_to_delete.append(m)

        # discard this bbox

        out = np.delete(out, rows_to_delete, axis=0)
        box_classes = np.delete(box_classes, rows_to_delete, axis=0)

        if out.shape[0] != 0:
            newboxes = out
            newclasses = box_classes
        else:
            newboxes= np.array([[0,0,0,0]])
            newclasses = np.array([0])
            # check whether there are any bboxes on the image, if not, discard
        newimg, new_bboxes, new_classes = aug.check_bbox_validity(newimg, newboxes, newclasses)
        if len(new_bboxes) != 0:

            images[k] = newimg
            total_boxes[k] = new_bboxes
            total_classes[k] = new_classes
 
            print("processing round: ", k)
            k = k+1

    # only retain k images
    final_aug_num = len(total_boxes)
    print('final number: ',final_aug_num )
    images = images[0:final_aug_num]

    return images.astype(np.uint8),total_boxes,total_classes







# added this function to chip with uids retained
# this function to discard bboxes that cut off to have less than 30 pixels in w/h 
def chip_image_with_uid(img,coords,classes,uids, shape=(300,300)):
    """
    Chip an image and get relative coordinates and classes.  Bounding boxes that pass into
        multiple chips are clipped: each portion that is in a chip is labeled. For example,
        half a building will be labeled if it is cut off in a chip. If there are no boxes,
        the boxes array will be [[0,0,0,0]] and classes [0].
        Note: This chip_image method is only tested on xView data-- there are some image manipulations that can mess up different images.
    Args:
        img: the image to be chipped in array format
        coords: an (N,4) array of bounding box coordinates for that image
        classes: an (N,1) array of classes for each bounding box
        shape: an (W,H) tuple indicating width and height of chips
    Output:
        An image array of shape (M,W,H,C), where M is the number of chips,
        W and H are the dimensions of the image, and C is the number of color
        channels.  Also returns boxes and classes dictionaries for each corresponding chip.
    """
    height,width,_ = img.shape
    wn,hn = shape
    
    w_num,h_num = (int(width/wn),int(height/hn))
    images = np.zeros((w_num*h_num,hn,wn,3))
    total_boxes = {}
    total_classes = {}
    total_uids = {}
    
    # debug
    threshold = 30  # threshold of # of pixels to discard bbox
    
    k = 0
    for i in range(w_num):
        for j in range(h_num):
            x = np.logical_or( np.logical_and((coords[:,0]<((i+1)*wn)),(coords[:,0]>(i*wn))),
                               np.logical_and((coords[:,2]<((i+1)*wn)),(coords[:,2]>(i*wn))))
            out = coords[x]
            y = np.logical_or( np.logical_and((out[:,1]<((j+1)*hn)),(out[:,1]>(j*hn))),
                               np.logical_and((out[:,3]<((j+1)*hn)),(out[:,3]>(j*hn))))
            outn = out[y]
            out = np.transpose(np.vstack((np.clip(outn[:,0]-(wn*i),0,wn),
                                          np.clip(outn[:,1]-(hn*j),0,hn),
                                          np.clip(outn[:,2]-(wn*i),0,wn),
                                          np.clip(outn[:,3]-(hn*j),0,hn))))
            
            box_classes = classes[x][y]
            box_uids = uids[x][y]
            
            # debug
            # remove bboxes that only have less than 20 pixels in w/h left in the image
            # only loop through ones that have 0 or wn/hn in the 4 coordinates
            rows_to_delete = list()
            for m in range(out.shape[0]):
                if(np.any([out[m] == 0]) or np.any([out[m] == wn]) or np.any([out[m] == hn])):
                    # see whether the width of bbox is less than 10 pixels?
                    bbox_w = out[m][2] - out[m][0]
                    bbox_h = out[m][3] - out[m][1]
                    if bbox_w < threshold or bbox_h < threshold:
                        rows_to_delete.append(m)
                        
            # discard this bbox
        
            out = np.delete(out, rows_to_delete, axis=0)
            box_classes = np.delete(box_classes, rows_to_delete, axis=0)
            box_uids = np.delete(box_uids, rows_to_delete, axis=0)
            
            
            if out.shape[0] != 0:
                total_boxes[k] = out
                total_classes[k] = box_classes
                total_uids[k] = box_uids
            else:
                total_boxes[k] = np.array([[0,0,0,0]])
                total_classes[k] = np.array([0])
                total_uids[k] = np.array([0])
            
            chip = img[hn*j:hn*(j+1),wn*i:wn*(i+1),:3]
            images[k]=chip
            
            k = k + 1
    
    return images.astype(np.uint8),total_boxes,total_classes, total_uids




# # changed this function to discard bboxes that cut off to have less than 20 pixels in w/h 
def chip_image(img,coords,classes,shape=(300,300)):
    """
    Chip an image and get relative coordinates and classes.  Bounding boxes that pass into
        multiple chips are clipped: each portion that is in a chip is labeled. For example,
        half a building will be labeled if it is cut off in a chip. If there are no boxes,
        the boxes array will be [[0,0,0,0]] and classes [0].
        Note: This chip_image method is only tested on xView data-- there are some image manipulations that can mess up different images.

    Args:
        img: the image to be chipped in array format
        coords: an (N,4) array of bounding box coordinates for that image
        classes: an (N,1) array of classes for each bounding box
        shape: an (W,H) tuple indicating width and height of chips

    Output:
        An image array of shape (M,W,H,C), where M is the number of chips,
        W and H are the dimensions of the image, and C is the number of color
        channels.  Also returns boxes and classes dictionaries for each corresponding chip.
    """
    height,width,_ = img.shape
    wn,hn = shape
    
    w_num,h_num = (int(width/wn),int(height/hn))
    images = np.zeros((w_num*h_num,hn,wn,3))
    total_boxes = {}
    total_classes = {}
    

    # debug
    threshold = 30  # threshold of # of pixels to discard bbox

    k = 0
    for i in range(w_num):
        for j in range(h_num):
            x = np.logical_or( np.logical_and((coords[:,0]<((i+1)*wn)),(coords[:,0]>(i*wn))),
                               np.logical_and((coords[:,2]<((i+1)*wn)),(coords[:,2]>(i*wn))))
            out = coords[x]
            y = np.logical_or( np.logical_and((out[:,1]<((j+1)*hn)),(out[:,1]>(j*hn))),
                               np.logical_and((out[:,3]<((j+1)*hn)),(out[:,3]>(j*hn))))
            outn = out[y]
            out = np.transpose(np.vstack((np.clip(outn[:,0]-(wn*i),0,wn),
                                          np.clip(outn[:,1]-(hn*j),0,hn),
                                          np.clip(outn[:,2]-(wn*i),0,wn),
                                          np.clip(outn[:,3]-(hn*j),0,hn))))
            box_classes = classes[x][y]
             # debug
            # remove bboxes that only have less than 20 pixels in w/h left in the image
            # only loop through ones that have 0 or wn/hn in the 4 coordinates
            rows_to_delete = list()
            for m in range(out.shape[0]):
                if(np.any([out[m] == 0]) or np.any([out[m] == wn]) or np.any([out[m] == hn])):
                    # see whether the width of bbox is less than 10 pixels?
                    bbox_w = out[m][2] - out[m][0]
                    bbox_h = out[m][3] - out[m][1]
                    if bbox_w < threshold or bbox_h < threshold:
                        rows_to_delete.append(m)
                        
            # discard this bbox
        
            out = np.delete(out, rows_to_delete, axis=0)
            box_classes = np.delete(box_classes, rows_to_delete, axis=0)            



            
            if out.shape[0] != 0:
                total_boxes[k] = out
                total_classes[k] = box_classes
            else:
                total_boxes[k] = np.array([[0,0,0,0]])
                total_classes[k] = np.array([0])
            
            chip = img[hn*j:hn*(j+1),wn*i:wn*(i+1),:3]
            images[k]=chip
            
            k = k + 1
    
    return images.astype(np.uint8),total_boxes,total_classes

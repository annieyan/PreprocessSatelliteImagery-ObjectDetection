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
"""

import numpy as np
from PIL import Image
#import tensorflow as tf
from PIL import Image, ImageDraw
import skimage.filters as filters
# for additional data augmentation
from skimage import data, exposure, img_as_float
import random
from skimage.transform import rescale, resize
from skimage.util import random_noise
import functools
from itertools import chain, combinations
from skimage.filters.rank import median
from skimage.morphology import disk
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

"""
Image augmentation utilities to be used for processing the dataset.  Importantly, these utilities modify
    the images as well as their respective bboxes (for example, in rotation).  Includes:
    rotation, shifting, salt-and-pepper, gaussian blurring.  Also includes a 'draw_bboxes' function
    for visualizing augmented images and bboxes
"""
# debug
# added additional aumentation choices for enlarging small classes
'''
Expand one training image into multiple ones through augmentation
The augmentation is agressive in that it uses all possible combinations of 
a list of augmentation methods.
Args:
        img: the image to be chipped in array format
        boxes: an (N,4) array of bounding box coordinates for that image
        classes: an (N,1) array of classes for each bounding box
        class_id: the class to be augmented
Output:
        An image array of shape (M,W,H,C), where M is the number of chips,
        W and H are the dimensions of the image, and C is the number of color
        channels.  Also returns boxes and classes dictionaries for each corresponding chip
'''
# https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string
# https://stackoverflow.com/questions/16739290/composing-functions-in-python 
def expand_aug_full(img, boxes, classes, class_id):
    func_list = ['change_brightness', 'change_contrast', 'vertical_flip', 'horizontal_flip', 'median_blur', 'zoomin']
    
    all_subset = all_subsets(func_list)
    #debug
    #print('locals, ', locals())
    # number of augmentation = # of all combinations
    # all_subset contains empty set, so -1
    num_aug = len(list(all_subset)) -1
    # debug
    #print('one image is augmented into: ', num_aug)
    # number of different choices of augmentation done to one image
    w,h, _ = img.shape
    images = np.zeros((num_aug,w,h,3))
    total_boxes = {}
    total_classes = {}

    #newimg =  np.copy(img)
    #newboxes = np.copy(boxes)
    #newclasses = np.copy(classes)

    k = 0   # k = [0, num_aug)
    for subset in  all_subsets(func_list):
        len_sub = len(subset)
        if len_sub == 0:
            continue
        newimg =  np.copy(img)
        newboxes = np.copy(boxes)
        newclasses = np.copy(classes)
        for idx in range(len_sub):
            if subset[idx] == 'zoomin':
                newimg, newboxes, newclasses = globals()[subset[idx]](newimg, newboxes, newclasses, class_id)
            else:
                newimg, newboxes, newclasses = globals()[subset[idx]](newimg, newboxes, newclasses)
        images[k] = newimg
        total_boxes[k] = newboxes
        total_classes[k] = newclasses
        # debug
        print("processing round: ", k)
        k = k+1

    return images.astype(np.uint8),total_boxes,total_classes



'''
Expand one training image into multiple ones (default 15) through augmentation
The augmentation uses random combinations of 
a list of augmentation methods.
Args:
        img: the image to be chipped in array format
        boxes: an (N,4) array of bounding box coordinates for that image
        classes: an (N,1) array of classes for each bounding box
        class_id: the class to be augmented
        num_aug: how many images to be generated 
Output:
        An image array of shape (M,W,H,C), where M is the number of chips,
        W and H are the dimensions of the image, and C is the number of color
        channels.  Also returns boxes and classes dictionaries for each corresponding chip
'''
# https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string
# https://stackoverflow.com/questions/16739290/composing-functions-in-python 
def expand_aug_random(img, boxes, classes, class_id, shift_or_not = False, num_aug = 15):
    if shift_or_not == False:

        func_list = ['change_brightness', 'change_contrast', 'vertical_flip', 'horizontal_flip', 'zoomin', 'rotate_image_and_boxes_formatted']  # 8
    else:
        func_list = ['change_brightness', 'change_contrast', 'vertical_flip', 'horizontal_flip', 'zoomin', 'rotate_image_and_boxes_formatted', 'shift_image_formatted']
    random_list = random_subsets(func_list, num_aug)
    #debug
    print('random functions to use: ', random_list)
    # debug
    w,h, _ = img.shape
    images = np.zeros((num_aug,w,h,3))
    total_boxes = {}
    total_classes = {}

    k = 0   # k = [0, num_augI
    #new_fun_list = list()  # choices of combinations of with fewer than 3 functions
    '''
    for subset in  all_subsets(func_list):
        len_sub = len(subset)
        if len_sub == 0:
            continue
        if len_sub > 3:
            continue
        new_fun_list.append(subset)

   
        # generate random 15 numbers from 0 to len(new_func_list)
        
    random_list = new_fun_list[np.random.randint(0,len(new_fun_list))]
    '''
    for func_comb in random_list:
        len_comb = len(func_comb)
        newimg =  np.copy(img)
        newboxes = np.copy(boxes)
        newclasses = np.copy(classes)
        for idx in range(len_comb):
            if func_comb[idx] == 'zoomin':
                newimg, newboxes, newclasses = globals()[func_comb[idx]](newimg, newboxes, newclasses, class_id)
            else:
                newimg, newboxes, newclasses = globals()[func_comb[idx]](newimg, newboxes, newclasses)
        
        newimg, new_bboxes, new_classes = check_bbox_validity(newimg, newboxes, newclasses)
        # check whether there are any bboxes on the image, if not, discard
        if len(new_bboxes) != 0:
      
            images[k] = newimg
            total_boxes[k] = new_bboxes
            total_classes[k] = new_classes
       # debug
            print("processing round: ", k)
            k = k+1

    # only retain k images
    final_aug_num = len(total_boxes)
    print('final augmentation number: ',final_aug_num )
    images = images[0:final_aug_num]

    return images.astype(np.uint8),total_boxes,total_classes


# check and delete invalid bboxes for an image
# check for all 0s, NaNs, and out of bounds 
def check_bbox_validity(img, bboxes, classes):
    w = img.shape[0]
    h = img.shape[1]
    bbox_len = len(bboxes)
    new_bboxes = []
    new_classes = []
    # for each bbox in that image, check and delete if not valid
    for i in range(bbox_len): 
        xmin, ymin, xmax, ymax = bboxes[i]
        # check for all 0
        if np.all(bboxes[i]==0):
            print('BBOX ALL ZEROS')
            continue
        # check for any NaNs
        if np.isnan(bboxes[i]).any():
            print('BBOXE HAS NANs')
            continue
        # check for out of bounds
        if xmin < 0 or ymin < 0 or xmax > w and ymax > h:
            print('BBOX out of bounds')
            continue
        # check for bbox len and width
        if (xmax - xmin) <= 0 or (ymax - ymin) <= 0:
            print('BBOX invalid height or width')
            continue
        newbox = np.array([xmin, ymin, xmax, ymax]).astype(np.int64)
        new_bboxes.append(newbox) 
        new_classes.append(classes[i])
    return img, new_bboxes, new_classes




def random_subsets(lst, num_aug):
    #func_list = ['change_brightness', 'change_contrast', 'vertical_flip', 'horizontal_flip', 'zoomin', 'gaussian_blur_formatted', 'shift_image_formatted', 'rotate_image_and_boxes_formatted']  # 8

    #all_subset = all_subsets(lst)
    #debug
    #print('locals, ', locals())
    # number of augmentation = # of all combinations
    # all_subset contains empty set, so -1
    #num_aug = len(list(all_subset)) -1
    # debug
    #print('one image is augmented into: ', num_aug)
    # number of different choices of augmentation done to one image
    new_fun_list = list()  # choices of combinations of with fewer than 3 functions
    for subset in  all_subsets(lst):
        len_sub = len(subset)
        if len_sub == 0:
            continue
        if len_sub > 3:
            continue
        new_fun_list.append(subset)
    # debug
    # manually added "gaussian blur"
    new_fun_list.append(('gaussian_blur_formatted',))
    new_fun_list.append(('gaussian_blur_formatted', 'horizontal_flip'))
    new_fun_list.append(('gaussian_blur_formatted', 'vertical_flip'))
    #new_fun_list.append(('gaussian_blur_formatted', 'shift_image_formatted'))
    new_fun_list.append(('gaussian_blur_formatted', 'rotate_image_and_boxes_formatted'))

        # generate random 15 numbers from 0 to len(new_func_list)
    new_fun_arr = np.array(new_fun_list)
    random_list = new_fun_arr[random.sample(range(0,len(new_fun_list)), num_aug)]
    return list(random_list)



# generate all possible combinations of sublist of a list 
def all_subsets(lst):
    return chain(*map(lambda x: combinations(lst, x), range(0, len(lst)+1)))



# gamma correction: see http://scikimgt-image.org/docs/dev/api/skimage.exposure.html 
# For gamma greater than 1, the histogram will shift towards left and the output image will be darker than the input image.
def change_brightness(img, boxes, classes):

    newimg = np.copy(img)
    #image = img_as_float(newimg)
    gamma_val  = random.uniform(0.7,1.2)
    newimg = exposure.adjust_gamma(img, gamma=gamma_val, gain=0.9)
    return newimg, boxes, classes


def change_contrast(img, boxes, classes):
    newimg = np.copy(img)
    v_min, v_max = np.percentile(img, (0.2, 99.8))
    newimg= exposure.rescale_intensity(img, in_range=(v_min, v_max))

    return newimg, boxes, classes



def vertical_flip(img,  boxes, classes):
    pivot = np.array([int(img.shape[0]/2),int(img.shape[1]/2)])
    newimg, newboxes = rotate_image_and_boxes(img, 180, pivot, boxes)
    return newimg, newboxes, classes


def horizontal_flip(img,  boxes, classes):
    newimg = img[:, ::-1]
    # flip boxes
    pivot = np.array([int(img.shape[0]/2),int(img.shape[1]/2)])
    newboxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        #The 'x' values are not centered by the x-center (shape[0]/2)
        #but rather the y-center (shape[1]/2)

        xmin = 2 * pivot[1] - xmin
        xmax = 2 * pivot[1] - xmax
        
        #ymax -= pivot[0]
        '''
        bfull = np.array([ [xmin,xmin,xmax,xmax] , [ymin,ymax,ymin,ymax]])
        c = np.dot(R,bfull)
        c[0] += pivot[1]
        c[0] = np.clip(c[0],0,img.shape[1])
        c[1] += pivot[0]
        c[1] = np.clip(c[1],0,img.shape[0])

        if np.all(c[1] == img.shape[0]) or np.all(c[1] == 0):
            c[0] = [0,0,0,0]
        if np.all(c[0] == img.shape[1]) or np.all(c[0] == 0):
            c[1] = [0,0,0,0]

        newbox = np.array([np.min(c[0]),np.min(c[1]),np.max(c[0]),np.max(c[1])]).astype(np.int64)

        if not (np.all(c[1] == 0) and np.all(c[0] == 0)):
            newboxes.append(newbox)
        '''
        newbox = np.array([xmin, ymin, xmax, ymax]).astype(np.int64)
        newboxes.append(newbox)
    return newimg, newboxes, classes

'''
# scale img to original input size, say 500 x 500
def image_scale(img, scale):
    image_rescaled = rescale(img, scale)
    return image_rescaled
'''

# todo: add exception handling
# zoom in areas of interest, say a particular area that containing bounding boxes
# of intersted classes, crop the region 
# and rescale to original size
def zoomin(img, boxes, classes, class_id):
    # locate a bouding box that contains class_id
    rand  = random.uniform(0.1,5)
    randint = random.randint(1, 10)
    # get a list of bbox indexes of things that of interest
    ind_list = [i for i, x in enumerate(classes) if str(x) == str(class_id) or x == class_id]
    # zoom in a certain scale: 0.5 - 1 of original size
    w = img.shape[0]
    h = img.shape[1]
    # when no index found, gives warning, and crop the center of the image 
    #assert len(ind_list)!=0, "class not in this chip"
    # debug
    threshold = 20  # threshold of # of pixels to discard bbox

    if len(ind_list) ==0:
        #raise ValueError("class not in this chip")
        print('class not in this chip, clipping center part of the chip')
        bbox_x_center = w/2
        bbox_y_center = h /2
    else:
    # randomly get one object to zoom
        ind = np.random.choice(range(len(ind_list)))
         # get bbox center:
        xmin, ymin, xmax, ymax = boxes[ind]
        bbox_x_center = (xmin + xmax)/2
        bbox_y_center = (ymin + ymax) /2    


    scale = random.uniform(0.5,1)
    #scale = 0.5
    scaled_x = w * scale  # cropped area width
    scaled_y = h * scale  # cropped area height
    

    # force the crop to be square and contain the chosen bbox
    if bbox_x_center < 1/2 * w:
         # start from leftmost
        startx = 0
        endx = int(startx + scaled_x)
        # should consider the case: if endx < xmax
        # but in harvey's case, trash heaps/bridges/ roads won't be larger than 150 pixel
    else:
        endx = w
        startx = int(w - scaled_x)
    if bbox_y_center < 1/2 * h:
        starty = 0
        endy = int(starty + scaled_y)
    else:
        endy = h
        starty = int(h - scaled_y)
    
    # crop a region, centered on the bbox of that ind
    '''
    startx =  int(w - scaled_x/2) if (w - scaled_x/2) > 0 else 0
    starty = int(h - scaled_y/2) if (h - scaled_y/2) > 0 else 0
    endx = w if (w + scaled_x/2) > w else int(w + scaled_x/2)   
    endy = h if (h + scaled_y/2) > h else int(h + scaled_y/2)
    '''
    


    # force the crop to be square
    '''
    actual_x = endx - startx
    actual_y = endy - starty
    scaled_x = min(actual_x, actual_y)
    scaled_y = scaled_x
    endx = startx
    '''
    newimg = img[startx: endx, starty: endy]
    image_rescaled = resize(newimg, (w, h), preserve_range=True)
    # debug
    #print('startx, endx', startx, endx)    
    #print('starty, endy', starty, endy)
    # scale bounding boxes inside the cropped area
    # see https://github.com/DIUx-xView/data_utilities/blob/master/wv_util.py
    newboxes = []
    newclasses = []
    # crop bboxes and scale them 
    # TODO: (boxes[:,0]<endy) should all turned into (boxes[:,0]<endx)???

    boxes = np.array(boxes)  # change to np array, otherwise, boxes[:,0] cannot access list
    x = np.logical_or( np.logical_and( (boxes[:,0]<endy),  (boxes[:,0]>starty)),
                               np.logical_and((boxes[:,2]<endy),  (boxes[:,2]>starty)))
    out = boxes[x]
    y = np.logical_or( np.logical_and(  (out[:,1]<endx),  (out[:,1]>startx)),
                               np.logical_and((out[:,3]<endx),  (out[:,3]>startx)))
    outn = out[y]    
    # TODO: out = np.transpose(np.vstack((np.clip(outn[:,0]-starty,0,scaled_y),
    # should be changed to np.clip(outn[:,0]-startx,0,scaled_x ???
    out = np.transpose(np.vstack((np.clip(outn[:,0]-starty,0,scaled_y),
                                          np.clip(outn[:,1]-startx,0, scaled_x),
                                          np.clip(outn[:,2]-starty,0,scaled_y),
                                          np.clip(outn[:,3]-startx,0, scaled_x))))
    box_classes = classes[x][y]


 # debug
    # remove bboxes that only have less than 20 pixels in w/h left in the image
    # only loop through ones that have 0 or wn/hn in the 4 coordinates
    rows_to_delete = list()
    for m in range(out.shape[0]):
        if(np.any([out[m] == 0]) or np.any([out[m] == scaled_x]) or np.any([out[m] == scaled_y])):
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

    # now rescale each bouding boxes
    
    enlarge_boxes = []
    for box in newboxes:
        xmin, ymin, xmax, ymax = box
        xmin = xmin * (1/scale)
        ymin = ymin * (1/scale)
        xmax = xmax * (1/scale)
        ymax = ymax * (1/scale)
        newbox = np.array([xmin, ymin, xmax, ymax]).astype(np.int64)
        enlarge_boxes.append(newbox)
    

    return image_rescaled,enlarge_boxes,  newclasses



# add gaussian noises
# should avoid producing negative pixel values
def gaussian_noise(img, boxes, classes):
    #newimg = random_noise(img, mode='gaussian', seed=None, clip=True, **kwargs)
    mean = 0
    var = 1
    noise = np.random.normal(mean, var,
                                 img.shape)
    newimg = img + noise
    #newimg[newimg < 0] = 0
    #newimg = newimg.clip(0)
    newimg = np.array(newimg.clip(0)).astype(np.uint8)
    return newimg, boxes, classes



def median_blur(img, boxes, classes):
    disk_size = np.random.choice(range(2, 6))
    newimg = np.copy(img)
    for i in range(img.shape[2]):
        newimg[:,:,i] = median(img[:,:,i], disk(disk_size))
    return newimg, boxes, classes    



# reload rotate_image_and_boxes function to fit in chaining of 
# a series of augmentation functions
def rotate_image_and_boxes_formatted(img, boxes, classes):
    """
    Rotates an image and corresponding bounding boxes.  Bounding box rotations are kept axis-aligned,
        so multiples of non 90-degrees changes the area of the bounding box.
    Args:
        img: the image to be rotated in array format
        deg: an integer representing degree of rotation
        pivot: the axis of rotation. By default should be the center of an image, but this can be changed.
        boxes: an (N,4) array of boxes for the image
    Output:
        Returns the rotated image array along with correspondingly rotated bounding boxes
    """

 #   if deg < 0:
  #      deg = 360-deg
  #  deg = int(deg)
        
   # angle = 360-deg
    
   # randomly generate degree 
    deg = np.random.randint(-10,10)
    angle = 360 - deg
    pivot = np.array([int(img.shape[0]/2),int(img.shape[1]/2)])



    padX = [img.shape[0] - pivot[0], pivot[0]]
    padY = [img.shape[1] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX, [0,0]], 'constant').astype(np.uint8)
    #scipy ndimage rotate takes ~.7 seconds
    #imgR = ndimage.rotate(imgP, angle, reshape=False)
    #PIL rotate uses ~.01 seconds
    imgR = Image.fromarray(imgP).rotate(angle)
    imgR = np.array(imgR)
    
    theta = deg * (np.pi/180)
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    #  [(cos(theta), -sin(theta))] DOT [xmin, xmax] = [xmin*cos(theta) - ymin*sin(theta), xmax*cos(theta) - ymax*sin(theta)]
    #  [sin(theta), cos(theta)]        [ymin, ymax]   [xmin*sin(theta) + ymin*cos(theta), xmax*cos(theta) + ymax*cos(theta)]

    newboxes = []
    # debug
    new_classes = []
    for i in range(len(boxes)):
        box = boxes[i]
        xmin, ymin, xmax, ymax = box
        #The 'x' values are not centered by the x-center (shape[0]/2)
        #but rather the y-center (shape[1]/2)
        
        xmin -= pivot[1]
        xmax -= pivot[1]
        ymin -= pivot[0]
        ymax -= pivot[0]

        bfull = np.array([ [xmin,xmin,xmax,xmax] , [ymin,ymax,ymin,ymax]])
        c = np.dot(R,bfull) 
        c[0] += pivot[1]
        c[0] = np.clip(c[0],0,img.shape[1])
        c[1] += pivot[0]
        c[1] = np.clip(c[1],0,img.shape[0])
        
        if np.all(c[1] == img.shape[0]) or np.all(c[1] == 0):
            c[0] = [0,0,0,0]
        if np.all(c[0] == img.shape[1]) or np.all(c[0] == 0):
            c[1] = [0,0,0,0]

        newbox = np.array([np.min(c[0]),np.min(c[1]),np.max(c[0]),np.max(c[1])]).astype(np.int64)

        if not (np.all(c[1] == 0) and np.all(c[0] == 0)):
            newboxes.append(newbox)
            new_classes.append(classes[i])
    
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]], newboxes, new_classes



# debug
# reload shift image function to be ready for chaining
# note: bboxes may change due to the shift
# therefore, classes were modified accordingly
def shift_image_formatted(image,bbox, classes):
    """
    Shift an image by a random amount on the x and y axis drawn from discrete  
        uniform distribution with parameter min(shape/10)
    Args:
        image: the image to be shifted in array format
        bbox: an (N,4) array of boxes for the image
    Output:
        The shifted image and corresponding boxes
    """
    shape = image.shape[:2]
    # debug
    # do more aggressive shift
    #maxdelta = min(shape)/10
    maxdelta = min(shape)/5
    dx,dy = np.random.randint(-maxdelta,maxdelta,size=(2))
    newimg = np.zeros(image.shape,dtype=np.uint8)
    
    nb = []
    new_classes = []
    bbox_count = len(bbox)
    # debug
    #i = 0 # index of box in bbox
    for i in range(len(bbox)):
        box  = bbox[i]
        xmin,xmax = np.clip((box[0]+dy,box[2]+dy),0,shape[1])
        ymin,ymax = np.clip((box[1]+dx,box[3]+dx),0,shape[0])

        #we only add the box if they are not all 0
        if not(xmin==0 and xmax ==0 and ymin==0 and ymax ==0):
            nb.append([xmin,ymin,xmax,ymax])
            new_classes.append(classes[i])

    newimg[max(dx,0):min(image.shape[0],image.shape[0]+dx),
           max(dy,0):min(image.shape[1],image.shape[1]+dy)] = \
    image[max(-dx,0):min(image.shape[0],image.shape[0]-dx),
          max(-dy,0):min(image.shape[1],image.shape[1]-dy)]
    
    threshold = 30
    w = newimg.shape[0]
    h = newimg.shape[1]
    # debug
    # remove bboxes that only have less than 20 pixels in w/h left in the image
    # only loop through ones that have 0 or wn/hn in the 4 coordinates
    rows_to_delete = list()
    nb = np.array(nb)
    for m in range(nb.shape[0]):
        if(np.any([nb[m] == 0]) or np.any([nb[m] == w ]) or np.any([nb[m] == h])):
         # see whether the width of bbox is less than 10 pixels?
            bbox_w = nb[m][2] - nb[m][0]
            bbox_h = nb[m][3] - nb[m][1]
            if bbox_w < threshold or bbox_h < threshold:
                rows_to_delete.append(m)

    # discard this bbox

    nb = np.delete(nb, rows_to_delete, axis=0)
    new_classes = np.delete(new_classes, rows_to_delete, axis=0)




    return newimg, nb, new_classes


# reloaded gaussian blue
def gaussian_blur_formatted(img, bboxes, classes):
    """
    Use a gaussian filter to blur an image
    Args:
        img: image to be augmented in array format
        max_sigma: the maximum variance for gaussian blurring
    Output:
        Augmented image
    """
    max_sigma=1
    new_img = img.clip(0, 255)
    return filters.gaussian(new_img,np.random.random()*max_sigma,multichannel=True)*255, bboxes,classes






def rotate_image_and_boxes(img, deg, pivot, boxes):
    """
    Rotates an image and corresponding bounding boxes.  Bounding box rotations are kept axis-aligned,
        so multiples of non 90-degrees changes the area of the bounding box.

    Args:
        img: the image to be rotated in array format
        deg: an integer representing degree of rotation
        pivot: the axis of rotation. By default should be the center of an image, but this can be changed.
        boxes: an (N,4) array of boxes for the image

    Output:
        Returns the rotated image array along with correspondingly rotated bounding boxes
    """

    if deg < 0:
        deg = 360-deg
    deg = int(deg)
        
    angle = 360-deg
    padX = [img.shape[0] - pivot[0], pivot[0]]
    padY = [img.shape[1] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX, [0,0]], 'constant').astype(np.uint8)
    #scipy ndimage rotate takes ~.7 seconds
    #imgR = ndimage.rotate(imgP, angle, reshape=False)
    #PIL rotate uses ~.01 seconds
    imgR = Image.fromarray(imgP).rotate(angle)
    imgR = np.array(imgR)
    
    theta = deg * (np.pi/180)
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    #  [(cos(theta), -sin(theta))] DOT [xmin, xmax] = [xmin*cos(theta) - ymin*sin(theta), xmax*cos(theta) - ymax*sin(theta)]
    #  [sin(theta), cos(theta)]        [ymin, ymax]   [xmin*sin(theta) + ymin*cos(theta), xmax*cos(theta) + ymax*cos(theta)]

    newboxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        #The 'x' values are not centered by the x-center (shape[0]/2)
        #but rather the y-center (shape[1]/2)
        
        xmin -= pivot[1]
        xmax -= pivot[1]
        ymin -= pivot[0]
        ymax -= pivot[0]

        bfull = np.array([ [xmin,xmin,xmax,xmax] , [ymin,ymax,ymin,ymax]])
        c = np.dot(R,bfull) 
        c[0] += pivot[1]
        c[0] = np.clip(c[0],0,img.shape[1])
        c[1] += pivot[0]
        c[1] = np.clip(c[1],0,img.shape[0])
        
        if np.all(c[1] == img.shape[0]) or np.all(c[1] == 0):
            c[0] = [0,0,0,0]
        if np.all(c[0] == img.shape[1]) or np.all(c[0] == 0):
            c[1] = [0,0,0,0]

        newbox = np.array([np.min(c[0]),np.min(c[1]),np.max(c[0]),np.max(c[1])]).astype(np.int64)

        if not (np.all(c[1] == 0) and np.all(c[0] == 0)):
            newboxes.append(newbox)
    
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]], newboxes

def shift_image(image,bbox):
    """
    Shift an image by a random amount on the x and y axis drawn from discrete  
        uniform distribution with parameter min(shape/10)

    Args:
        image: the image to be shifted in array format
        bbox: an (N,4) array of boxes for the image

    Output:
        The shifted image and corresponding boxes
    """
    shape = image.shape[:2]
     # do more aggressive shift
    #maxdelta = min(shape)/10
    maxdelta = min(shape)/5
    dx,dy = np.random.randint(-maxdelta,maxdelta,size=(2))
    newimg = np.zeros(image.shape,dtype=np.uint8)
    
    nb = []
    for box in bbox:
        xmin,xmax = np.clip((box[0]+dy,box[2]+dy),0,shape[1])
        ymin,ymax = np.clip((box[1]+dx,box[3]+dx),0,shape[0])

        #we only add the box if they are not all 0
        if not(xmin==0 and xmax ==0 and ymin==0 and ymax ==0):
            nb.append([xmin,ymin,xmax,ymax])
    
    newimg[max(dx,0):min(image.shape[0],image.shape[0]+dx),
           max(dy,0):min(image.shape[1],image.shape[1]+dy)] = \
    image[max(-dx,0):min(image.shape[0],image.shape[0]-dx),
          max(-dy,0):min(image.shape[1],image.shape[1]-dy)]
    
    return newimg, nb


# todo: addativeGaussianNoise
def salt_and_pepper(img,prob=.005):
    """
    Applies salt and pepper noise to an image with given probability for both.

    Args:
        img: the image to be augmented in array format
        prob: the probability of applying noise to the image

    Output:
        Augmented image
    """

    newimg = np.copy(img)
    whitemask = np.random.randint(0,int((1-prob)*200),size=img.shape[:2])
    blackmask = np.random.randint(0,int((1-prob)*200),size=img.shape[:2])
    newimg[whitemask==0] = 255
    newimg[blackmask==0] = 0
        
    return newimg

# reduce max_sigma = 1.5 to 0.5
def gaussian_blur(img, max_sigma=1):
    """
    Use a gaussian filter to blur an image

    Args:
        img: image to be augmented in array format
        max_sigma: the maximum variance for gaussian blurring

    Output:
        Augmented image
    """
    return filters.gaussian(img,np.random.random()*max_sigma,multichannel=True)*255


# added 
# draw bboxes with bbox uid
# The text size is suitable for small images such as 256 * 256
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
    
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 15)
    #font = ImageFont.truetype('arial.ttf', 24)


    idx = 0

    for b in boxes:
        xmin,ymin,xmax,ymax = b
        
        for j in range(3):
            draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="red")
        draw.text((xmin+30, ymin+30), str(int(uids[idx])), font = font, fill = "orange")
        idx +=1
    return source




# draw bboxes with multiple classes with uids
# red for damaged buildings, and green for non-damaged buildings.
def draw_bboxes_withindex_multiclass(img,boxes,classes, uids):
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
    
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 15)
    #font = ImageFont.truetype('arial.ttf', 24)
    idx = 0

    for b in boxes:
        xmin,ymin,xmax,ymax = b
        if classes[idx] == 1: 
        
            for j in range(3):
                draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="red")
        else:
            for j in range(3):
                draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="green")
        draw.text((xmin+30, ymin+30), str(int(uids[idx])), font = font, fill = "orange")
        idx +=1
    return source






def draw_bboxes(img,boxes):
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

    idx = 0

    for b in boxes:
        xmin,ymin,xmax,ymax = b
        
        for j in range(3):
            draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="red")
    return source

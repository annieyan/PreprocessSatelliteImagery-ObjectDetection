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



'''
This is for creating a multiclass training data for harvey hurricane
In the case of Digital Globe data, there are two classes: damaged buildings / non-damaged buildings
This file produce TF record for training data only.
Big tiffs (2048) are chipped sequentially into 200 x 200.
Augmentation will be applied to chips that contain damaged buildings (class1)
Additional augmentation is done by shifting small chips,
But shifting is done in big tiff, instead of leaving black pixels at the edge

Optional
Then randomly discard some chips that contain ONLY non-damaged buildings (class2) 
'''


from PIL import Image
import tensorflow as tf
import io
import glob
from tqdm import tqdm
import numpy as np
import logging
import argparse
import os
import json
import wv_util as wv
import tfr_util as tfr
import aug_util as aug
import csv

"""
  A script that processes xView imagery. 
  Args:
      image_folder: A folder path to the directory storing xView .tif files
        ie ("xView_data/")

      json_filepath: A file path to the GEOJSON ground truth file
        ie ("xView_gt.geojson")

      test_percent (-t): The percentage of input images to use for test set

      suffix (-s): The suffix for output TFRecord files.  Default suffix 't1' will output
        xview_train_t1.record and xview_test_t1.record

      augment (-a): A boolean value of whether or not to use augmentation

  Outputs:
    Writes two files to the current directory containing training and test data in
        TFRecord format ('xview_train_SUFFIX.record' and 'xview_test_SUFFIX.record')
"""




def detect_blackblock(img):
    # check the # of pixels that with RGB values are all equal to 0
    w,h,c = img.shape
    black_pixel_count=0
    threshold = 0.9 * w * h * 3
    non_black_count = np.count_nonzero(img)
    if non_black_count > threshold:
        return False
    else:
        return True



def detect_clouds(img,  boxes, classes):
    mean_threshold_min = 160
    w, h, _ = img.shape
    #print('w,h', w, h)
    var_threshold = 18
    rows_to_delete = list()
    boxes = np.array(boxes)
    for i in range(boxes.shape[0]):
        xmin, ymin, xmax, ymax = boxes[i]
#         ymin = 0
        if xmin < 0:
            xmin = 0
        if ymin<0:
            y_min = 0
        if xmax > h:
            print('xmax > h')
            xmax = h
        if ymax > w:
            print('ymax > w')
            ymax= h
        #print(xmin, ymin, xmax, ymax)
        cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]  # note the order of w/h
        array_img =  np.array(cropped_img)
        mean_img = np.mean(array_img)
        
        var_img = np.std(array_img)
        #print('var_img',var_img, i)
        if var_img < var_threshold and mean_img > mean_threshold_min:
            print('bounding box i has cloud', i)
            # need to delete this bbox
            rows_to_delete.append(i)
    print('rows_to_delete',rows_to_delete)
            
    if len(rows_to_delete) == 0:
        classes = np.array(classes)
        return img,  boxes, classes
    else:
        # return boxes and classes with clouds removed
        new_coords = np.delete(boxes, rows_to_delete, axis=0)
        new_classes = np.delete(classes, rows_to_delete, axis=0)
        return img, new_coords, new_classes
        
        
    



def get_images_from_filename_array(coords,chips,classes,folder_names,res=(250,250)):
    """
    Gathers and chips all images within a given folder at a given resolution.

    Args:
        coords: an array of bounding box coordinates
        chips: an array of filenames that each coord/class belongs to.
        classes: an array of classes for each bounding box
        folder_names: a list of folder names containing images
        res: an (X,Y) tuple where (X,Y) are (width,height) of each chip respectively

    Output:
        images, boxes, classes arrays containing chipped images, bounding boxes, and classes, respectively.
    """

    images =[]
    boxes = []
    clses = []

    k = 0
    bi = 0   
    
    for folder in folder_names:
        fnames = glob.glob(folder + "*.tif")
        fnames.sort()
        for fname in tqdm(fnames):
            #Needs to be "X.tif" ie ("5.tif")
            name = fname.split("\\")[-1]
            arr = wv.get_image(fname)
            
            img,box,cls = wv.chip_image(arr,coords[chips==name],classes[chips==name],res)

            for im in img:
                images.append(im)
            for b in box:
                boxes.append(b)
            for c in cls:
                clses.append(cls)
            k = k + 1
            
    return images, boxes, clses

def shuffle_images_and_boxes_classes(im,box,cls):
    """
    Shuffles images, boxes, and classes, while keeping relative matching indices

    Args:
        im: an array of images
        box: an array of bounding box coordinates ([xmin,ymin,xmax,ymax])
        cls: an array of classes

    Output:
        Shuffle image, boxes, and classes arrays, respectively
    """
    assert len(im) == len(box)
    assert len(box) == len(cls)
    
    perm = np.random.permutation(len(im))
    out_b = {}
    out_c = {}
    
    k = 0 
    for ind in perm:
        out_b[k] = box[ind]
        out_c[k] = cls[ind]
        k = k + 1
    return im[perm], out_b, out_c

'''
Datasets
_multires: multiple resolutions. Currently [(500,500),(400,400),(300,300),(200,200)]
_aug: Augmented dataset
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder", help="Path to folder containing image chips (ie 'Image_Chips/' ")
    parser.add_argument("json_filepath", help="Filepath to GEOJSON coordinate file")
    parser.add_argument("-t", "--test_percent", type=float, default=0.333,
                    help="Percent to split into test (ie .25 = test set is 25% total)")
    parser.add_argument("-s", "--suffix", type=str, default='t1',
                    help="Output TFRecord suffix. Default suffix 't1' will output 'xview_train_t1.record' and 'xview_test_t1.record'")
    parser.add_argument("-a","--augment", type=bool, default=False,
    				help="A boolean value whether or not to use augmentation")
    # debug: added percent of data to produce, the purpose is to produce small dataset for fast algorithm development
    parser.add_argument("-p", "--sample_percent", type=int, default = 1, help = "Portion to sample data (1/sample_percent) from the original dataset. Meaning that only use a portion of the dataset to construct training and testing. The purpose is for fast algorithm development")
    


    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    #resolutions should be largest -> smallest.  We take the number of chips in the largest resolution and make
    #sure all future resolutions have less than 1.5times that number of images to prevent chip size imbalance.
    #res = [(500,500),(400,400),(300,300),(200,200)]
    #res = [(300,300)]
    #res = [(512,512)]
    
    res = [(200,200)]


    AUGMENT = args.augment
    # debug
    SAVE_IMAGES = False
    #SAVE_IMAGES = True
    images = {}
    boxes = {}
    train_chips = 0
    #test_chips = 0


    num_class1_bbox = 0   # num of bbox of class1
    num_class2_bbox = 0
    num_class1_chip = 0   # number of chips contain class 1
    num_class2_chip = 0
    num_class1_aug_bbox = 0  # number of augmentated bbox
    num_class2_aug_bbox = 0
    num_class1_aug_chip = 0
    num_class2_aug_chip = 0
    num_shifted_chips = 0 # number of chips added by shift
    #Parameters
    max_chips_per_res = 100000
    train_writer = tf.python_io.TFRecordWriter("harvey_train_%s.record" % args.suffix)
    #test_writer = tf.python_io.TFRecordWriter("harvey_test_%s.record" % args.suffix)

    #coords,chips,classes = wv.get_labels(args.json_filepath)
    coords,chips,classes,uids = wv.get_labels_w_uid_nondamaged(args.json_filepath)


    # debug
    sample_percent = args.sample_percent
    # a list of classes to be augment. Set to set to be empty if no augmentation
    # is wanted
    if AUGMENT == True:
 
        class_to_aug = set([1])
    else:
        class_to_aug = set([])
    num_aug_per_class = {}  # class_id: # of augmentation generated
    for class_id in class_to_aug:
        num_aug_per_class[class_id] = 0


    #debug
    # for cloud removing and black portion removing
    num_cloud_rm = 0  # number of 512 x 512 chips that have clouds removed
    num_black = 0  # number of 512 x 512 chips that have black parts 



    for res_ind, it in enumerate(res):
        tot_box = 0
        logging.info("Res: %s" % str(it))
        ind_chips = 0

        fnames = glob.glob(args.image_folder + "*.tif")
        fnames.sort()

        for fname in tqdm(fnames):
            #Needs to be "X.tif", ie ("5.tif")
            #Be careful!! Depending on OS you may need to change from '/' to '\\'.  Use '/' for UNIX and '\\' for windows
            name = fname.split("/")[-1]
            # debug
            #print('file name: ', name)
            arr = wv.get_image(fname)
            
            # debug
            print('file name: ', name)
            #print('classes[chips==name], ', classes[chips==name])

            im1,box1,classes_final1 = wv.chip_image(arr,coords[chips==name],classes[chips==name],it)
            

#Shuffle images & boxes all at once. Comment out the line below if you don't want to shuffle images
            im1,box1,classes_final1 = shuffle_images_and_boxes_classes(im1,box1,classes_final1)

            if AUGMENT:

            # do translation here / shift while chipping
                prob_shift = np.random.randint(5,10)

                im2,box2,classes_final2 = wv.random_crop_from_center(arr,coords[chips==name],classes[chips==name],  prob_shift, it)            
                sequetial_chip_len =  im1.shape[0]

                im = np.zeros((im1.shape[0]+ im2.shape[0],it[0], it[0],3))
                box = box1.copy()
                classes_final = classes_final1.copy()
            
                print('# of chips added by shift: ',  im2.shape[0])
                num_shifted_chips += im2.shape[0]
                for idx1, image in enumerate(im1):
                    im[idx1] = im1[idx1]
                for idx2, image in enumerate(im2):
                    im[im1.shape[0]+idx2] = im2[idx2]
                    box[im1.shape[0]+idx2] = box2[idx2]
                    classes_final[im1.shape[0]+idx2] = classes_final2[idx2]
                im = im.astype(np.uint8)
            else:
                im = im1.copy()
                box = box1.copy()
                classes_final = classes_final1.copy()

            #Shuffle images & boxes all at once. Comment out the line below if you don't want to shuffle images
            #im,box,classes_final = shuffle_images_and_boxes_classes(im,box,classes_final)
           # split_ind = int(im.shape[0] * args.test_percent)
            
                if idx%sample_percent !=0:
                    continue
                 # debug
                print('processing idx: ', idx)
                # debug
                # remove black block
                if detect_blackblock(image):
                    num_black +=1
                    continue
                # remove clouds
                image, new_coords, new_classes = detect_clouds(image,box[idx],classes_final[idx])                
                if len(new_coords)!= len(box[idx]):
                    num_cloud_rm += 1

                # debug: changed image,box[idx],classes_final[idx] to newly constructed img and box
                #tf_example = tfr.to_tf_example(image,box[idx],classes_final[idx])
               
                # debug
                # get statistics about number of damaged buildings and non-damaged buildings
                #print("type of new_classes: ", type(new_classes))
             #   print('new_classes', new_classes)
                #print('new_classes[new_classes==1]: ', new_classes.count(1))

                # number of class 1 bbox in the small chip
                local_class1 = new_classes[new_classes==1].shape[0]  
                local_class2 = new_classes[new_classes==2].shape[0]

            
                
                # debug
                # here only write into TF RECORD classes == 1
                #tf_example = tfr.to_tf_example(image, new_coords[new_classes ==1], new_classes[new_classes == 1])
                
                tf_example = tfr.to_tf_example(image, new_coords, new_classes)


                #Check to make sure that the TF_Example has valid bounding boxes.  
                #If there are no valid bounding boxes, then don't save the image to the TFRecord.
                float_list_value_xmin = tf_example.features.feature['image/object/bbox/xmin'].float_list.value
                float_list_value_ymin = tf_example.features.feature['image/object/bbox/ymin'].float_list.value
                float_list_value_xmax = tf_example.features.feature['image/object/bbox/xmax'].float_list.value
                float_list_value_ymax = tf_example.features.feature['image/object/bbox/ymax'].float_list.value

                if (ind_chips < max_chips_per_res and np.array(float_list_value_xmin).any() and np.array(float_list_value_xmax).any() and np.array(float_list_value_ymin).any() and np.array(float_list_value_ymax).any()):
                    tot_box+=np.array(float_list_value_xmin).shape[0]

                    #debug
                    num_class1_bbox += local_class1
                    num_class2_bbox += local_class2
            
                    #if idx < split_ind:
                     #   test_writer.write(tf_example.SerializeToString())
                      #  test_chips+=1
                       # if SAVE_IMAGES:
                                    # debug: changed save dir
                            #debug
                            # draw only DAMAGED buildings
                            #aug.draw_bboxes(image, new_coords[new_classes ==1]).save('./harvey_ms_img_inspect_val_2class_noclean/img_%s_%s.png'%(name,str(idx)))
                        #    aug.draw_bboxes(image, new_coords).save('./harvey_ms_img_inspect_val_2class_noclean/img_%s_%s.png'%(name,str(idx)))
   
                    '''
                    # debug
                    # randomly discard chips that contain ONLY class2
                    if local_class1 == 0:
                        p = np.random.randint(0,10)
                        if p < 1:
                            print('discarding this chip that contains ONLY class2')
                            continue
                    '''

                    if local_class1 > 0:
                        num_class1_chip +=1
                    if local_class2 > 0:
                        num_class2_chip +=1


                    #else:
                    train_writer.write(tf_example.SerializeToString())
                    train_chips += 1
                    if SAVE_IMAGES and idx%5 ==0:
                                    # debug: changed save dir
                            #aug.draw_bboxes(image, new_coords[new_classes ==1]).save('./harvey_ms_img_inspect_train_2class_noclean/img_%s_%s.png'%(name,str(idx)))
                        aug.draw_bboxes(image, new_coords).save('./tomnod_2class_train_inspect/img_%s_%s.png'%(name,str(idx)))
     
                    ind_chips +=1
                    
                    # debug
                    # store the training and validation images with bboxes for inspection
                    '''
                    if SAVE_IMAGES:
                                    # debug: changed save dir
                        aug.draw_bboxes(image, new_coords).save('./harvey_img_inspect/img_%s_%s.png'%(name,str(idx)))
                    '''
                

                    #Make augmentation probability proportional to chip size.  Lower chip size = less chance.
                    #This makes the chip-size imbalance less severe.
                    #prob = np.random.randint(0,np.max(res))
                    #for 200x200: p(augment) = 200/500 ; for 300x300: p(augment) = 300/500 ...
                    prob = np.random.randint(0,50)


                    # debug
                    # added customized data augmentation for minor classes
                    #class_to_aug = [2, 3, 4]  # damaged roads, trash heaps, and bridges
                    # Minor classes will be augmented to 63 times larger with various augmentations
                    # 1. Detect whether minor classes are in the small chips, if yes, augment 
                    # this chip. The output will be a tensor of augmented images, bboxes, and classes
                    # unpack the output to tfrecord TRAINING data. 
                    # 2. If the chip does not contain any minor classes, go to normal augmentation
                    MINOR_CLASS_FLAG = False
                    for class_id in class_to_aug:
                        #num_aug_per_class[class_id] = 0
                        #num_aug_this_class = 0
                        # debug
                        # print('checking whether this chip contain class: ', class_id)
                        # this chip contains minor classes
                        #if np.any(classes_final[idx][:]== class_id):
                        #if class_id in set(classes_final[idx]) and idx > split_ind:
                        if class_id in set(new_classes) and AUGMENT == True:
                          #       skip_augmentation.add(idx)
                            MINOR_CLASS_FLAG = True
                         #   print('trying to call expand_aug for chip: ', idx)
                            #im_aug,boxes_aug,classes_aug= aug.expand_aug_random(image, box[idx], classes_final[idx], class_id)  
                            # debug
                            # added to TF RECORD damaged building only
                            #im_aug,boxes_aug,classes_aug= aug.expand_aug_random(image, new_coords[new_classes ==1], new_classes[new_classes==1], class_id)  

                            # debug
                            # augment only sequential chips, not shifted chips
                            if idx >= sequetial_chip_len:
                                continue
                            im_aug,boxes_aug,classes_aug= aug.expand_aug_random(image, new_coords, new_classes, class_id)

                            #debug
                            print('augmentig chip: ', idx)
                            num_aug = 0
                            for aug_idx, aug_image in enumerate(im_aug):
                                # debug
                                # added to record only damaged buidings
                                tf_example_aug = tfr.to_tf_example(aug_image, boxes_aug[aug_idx],classes_aug[aug_idx])

                                aug_local_num_class1 = classes_aug[aug_idx].count(1)
                                aug_local_num_class2 = classes_aug[aug_idx].count(2)


            
                                #Check to make sure that the TF_Example has valid bounding boxes.  
                #If there are no valid bounding boxes, then don't save the image to the TFRecord.
                                float_list_value_xmin = tf_example_aug.features.feature['image/object/bbox/xmin'].float_list.value
                                float_list_value_xmax = tf_example_aug.features.feature['image/object/bbox/xmax'].float_list.value
                                float_list_value_ymin = tf_example_aug.features.feature['image/object/bbox/ymin'].float_list.value
                                float_list_value_ymax = tf_example_aug.features.feature['image/object/bbox/ymax'].float_list.value

                                # debug
                                #num_aug = 0
                                if (np.array(float_list_value_xmin).any() and np.array(float_list_value_xmax).any() and np.array(float_list_value_ymin).any() and np.array(float_list_value_ymax).any()):
                                    tot_box+=np.array(float_list_value_xmin).shape[0]
                    
                                    train_writer.write(tf_example_aug.SerializeToString())
                                    num_aug = num_aug + 1
                                    train_chips+=1
                                    num_aug_per_class[class_id] = num_aug_per_class[class_id]+1
                         #           num_aug_this_class=num_aug_this_class + 1
                                    
                                    num_class1_aug_bbox += aug_local_num_class1
                                    num_class2_aug_bbox += aug_local_num_class2
                                    
                                    if aug_local_num_class1 > 0:
                                        num_class1_aug_chip += 1
                                    if aug_local_num_class2 > 0:
                                        num_class2_aug_chip += 1


                                    # debug
                                    if aug_idx%10 == 0 and SAVE_IMAGES:
                                    # debug: changed save dir
                                        aug_image = (aug_image).astype(np.uint8)
                                        aug.draw_bboxes(aug_image,boxes_aug[aug_idx]).save('./tomnod_aug_2class/img_aug_%s_%s_%s_%s.png'%(name, str(idx), str(aug_idx), str(class_id)))
                            # debug
                            print('augmenting class: ', class_id)
                            print('number of augmentation: ',num_aug)
                        #num_aug_per_class[class_id] = num_aug_this_class

                    # it: iterator for different resolutions
                    # The chunk below is DEPRECATED
                    # start to augment the rest
                    '''
                    if AUGMENT and prob < it[0] and MINOR_CLASS_FLAG == False:
                        
                        for extra in range(3):
                            center = np.array([int(image.shape[0]/2),int(image.shape[1]/2)])
                            deg = np.random.randint(-10,10)
                            #deg = np.random.normal()*30
                            # changed
                            # remove and gaussian blur
                            
                            newimg = aug.gaussian_blur(image)
                            #newimg = image

                            #.3 probability for each of shifting vs rotating vs shift(rotate(image))
                            p = np.random.randint(0,3)
                            # debug
                            # modified to use the removed cloud version of bboxes
                            # image, new_coords, new_classes
                            if p == 0:
                                newimg,nb = aug.shift_image(newimg,new_coords)
                                #newimg,nb = aug.shift_image(newimg,box[idx])
                            elif p == 1:
                                newimg,nb = aug.rotate_image_and_boxes(newimg,deg,center,new_coords)
                                #newimg,nb = aug.rotate_image_and_boxes(newimg,deg,center,box[idx])
                            elif p == 2:
                                newimg,nb = aug.rotate_image_and_boxes(newimg,deg,center,new_coords)
                                #newimg,nb = aug.rotate_image_and_boxes(newimg,deg,center,box[idx])
                                newimg,nb = aug.shift_image(newimg,nb)
                                

                            newimg = (newimg).astype(np.uint8)

                            if idx%100 == 0 and SAVE_IMAGES:
                                #debug
                                # changed save dir
                                Image.fromarray(newimg).save('./augmented_img_60/img_%s_%s_%s.png'%(name,extra,it[0]))

                            if len(nb) > 0:
                                # debug
                                # modified to use the cloud removed bboxs
                                tf_example = tfr.to_tf_example(newimg,nb,new_classes)
                                #tf_example = tfr.to_tf_example(newimg,nb,classes_final[idx])

                                #DonI't count augmented chips for chip indices
                                # changed
                                # removed data augmentation for test data
                                if idx < split_ind:
                                  #  test_writer.write(tf_example.SerializeToString())
                                   # test_chips += 1
                                    continue
                                else:
                                    train_writer.write(tf_example.SerializeToString())
                                    train_chips+=1
                            # debug:
                	    # save image + bounding boxes for debug
                            #else:
                                    if idx%100 ==0 and SAVE_IMAGES:
                                        # debug: changed save dir
                                        aug.draw_bboxes(newimg,nb).save('./harvey_augmented/img_aug_%s_%s_%s.png'%(name,extra,it[0]))
        '''
        ''' 
        # do translation here / shift while chipping
        prob_shift = np.random.randint(5,10)

        im2,box2,classes_final2 = wv.random_crop_from_center(arr,coords[chips==name],classes[chips==name],  prob_shift, it)

        #    im = np.zeros((im1.shape[0]+ im2.shape[0],it[0], it[0],3))
         #   box = box1.copy()
          #  classes_final = classes_final1.copy()

         print('# of chips added by shift: ',  im2.shape[0])
           # for idx1, image in enumerate(im1):
            #    im[idx1] = im1[idx1]
            #for idx2, image in enumerate(im2):
             #   im[im1.shape[0]+idx2] = im2[idx2]
              #  box[im1.shape[0]+idx2] = box2[idx2]
               # classes_final[im1.shape[0]+idx2] = classes_final2[idx2]
         im2 = im2.astype(np.uint8)
         '''





    if res_ind == 0:
        max_chips_per_res = int(ind_chips * 1.5)
        logging.info("Max chips per resolution: %s " % max_chips_per_res)

        logging.info("Tot Box: %d" % tot_box)
        logging.info("Chips: %d" % ind_chips)

    # debug
    for key, val in num_aug_per_class.items():
        print('for class:' , key)
        print('augmentation applied: ', val)
    # debug
    print('num of black small chips removed: ', num_black)
    print('num of small chips containing clouds:', num_cloud_rm)

    print('num of original class 1 bboxes: ', num_class1_bbox)
    print('num of original class 2 bboxes: ', num_class2_bbox)

    print('num of class 1 bbox augmented: ', num_class1_aug_bbox)
    print('num of class 2 bbox augmented: ', num_class2_aug_bbox)

    print('num of class 1 bbox in total: ', num_class1_aug_bbox + num_class1_bbox)
    print('num of class 2 bbox in total: ', num_class2_aug_bbox + num_class2_bbox)

 

    print('num of original chips that contain class 1: ', num_class1_chip)
    print('num of original chips that cntain class 2 bboxes: ', num_class2_chip)

    print('num of class 1 chips augmented: ', num_class1_aug_chip)
    print('num of class 2 chips augmented: ', num_class2_aug_chip)

    print('number of chips added by shift: ', num_shifted_chips)

    print('num of chips that contain class 1 bbox in total: ', num_class1_aug_chip + num_class1_chip)
    print('num of chips that contain class 2 bbox in total: ', num_class2_aug_chip + num_class2_chip)



    logging.info("saved: %d train chips" % train_chips)
    #logging.info("saved: %d test chips" % test_chips)
    train_writer.close()
    #test_writer.close() 

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
import tensorflow as tf
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tqdm import tqdm
import argparse
from det_util import generate_detections, generate_detection_for_single_image
import json

#import sys
#from os import path
#sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
#sys.path.append('../')
#from data_utilities.process_wv_ms import detect_blackblock
#from .wv_util import chip_image 
"""
Inference script to generate a file of predictions given an input.

Args:
    checkpoint: A filepath to the exported pb (model) file.
        ie ("saved_model.pb")

    chip_size: An integer describing how large chips of test image should be

    input: A filepath to a single test chip
        ie ("1192.tif")

    output: A filepath where the script will save  its predictions
        ie ("predictions.txt")


Outputs:
    Writes a file specified by the 'output' parameter containing predictions for the model.
        Per-line format:  xmin ymin xmax ymax class_prediction score_prediction
        Note that the variable "num_preds" is dependent on the trained model 
        (default is 250, but other models have differing numbers of predictions)

"""

def chip_image(img, chip_size=(300,300)):
    """
    Segmzent an image into NxWxH chips

    Args:
        img : Array of image to be chipped
        chip_size : A list of (width,height) dimensions for chips

    Outputs:
        An ndarray of shape (N,W,H,3) where N is the number of chips,
            W is the width per chip, and H is the height per chip.

    """
    width,height,_ = img.shape
    wn,hn = chip_size
    images = np.zeros((int(width/wn) * int(height/hn),wn,hn,3))
    k = 0
    for i in tqdm(range(int(width/wn))):
        for j in range(int(height/hn)):
            
        #    chip = img[wn*i:wn*(i+1),hn*j:hn*(j+1),:3]
            chip = img[hn*j:hn*(j+1), wn*i:wn*(i+1),:3]
            images[k]=chip
            
            k = k + 1
    
    return images.astype(np.uint8)





# # changed this function to discard bboxes that cut off to have less than 20 pixels in w/h 
def chip_image_withboxes(img,coords,shape=(300,300)):
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
    #total_classes = {}


    # debug
    # TODO: determine whether to discard bboxes at the edge
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
     #       box_classes = classes[x][y]
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
      #      box_classes = np.delete(box_classes, rows_to_delete, axis=0)


            if out.shape[0] != 0:
                total_boxes[k] = out
       #         total_classes[k] = box_classes
            else:
                total_boxes[k] = np.array([[0,0,0,0]])
        #        total_classes[k] = np.array([0])

            chip = img[hn*j:hn*(j+1),wn*i:wn*(i+1),:3]
            images[k]=chip
            k = k + 1

    return images.astype(np.uint8),total_boxes









def draw_bboxes(img,boxes,classes):
    """
    Draw bounding boxes on top of an image

    Args:
        img : Array of image to be modified
        boxes: An (N,4) array of boxes to draw, where N is the number of boxes.
        classes: An (N,1) array of classes corresponding to each bounding box.

    Outputs:
        An array of the same shape as 'img' with bounding boxes
            and classes drawn

    """
    source = Image.fromarray(img)
    draw = ImageDraw.Draw(source)
    w2,h2 = (img.shape[0],img.shape[1])

    idx = 0

    for i in range(len(boxes)):
        xmin,ymin,xmax,ymax = boxes[i]
        c = classes[i]

        draw.text((xmin+15,ymin+15), str(c))

        for j in range(4):
            draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="red")
    return source



# debug
# load building footprints from geojson file
# Detections will only be made in chips that contain any buildings
def load_building_footprint_nondamaged(fname):
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
#                 if b_id == '20170831_105001000B95E100_3020021_jpeg_compressed_06_01.tif':
#                     print('found chip!')
                bbox = data['features'][i]['properties']['bb'][1:-1].split(",")
                val = np.array([int(num) for num in data['features'][i]['properties']['bb'][1:-1].split(",")])

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
    coords = np.add(coords, add_np)

    return coords, chips





if __name__ == "__main__":
    # debug
    # added loading geojson to crop out chips with buildings
    parser = argparse.ArgumentParser()
    parser.add_argument("json_filepath",help="geojson file")
    parser.add_argument("-c","--checkpoint", default='pbs/model.pb', help="Path to saved model")
    parser.add_argument("-cs", "--chip_size", default=300, type=int, help="Size in pixels to chip input image")
    parser.add_argument("input", help="Path to test chip")
    parser.add_argument("-o","--output",default="predictions.txt",help="Filepath of desired output")
    args = parser.parse_args()

    #Parse and chip images
    arr = np.array(Image.open(args.input))
    chip_size = (args.chip_size,args.chip_size)
    #images = chip_image(arr,chip_size)
    #print(images.shape)
    image_name = args.input.split("/")[-1]

    # TODO: loading building footprints, only make detections with the chips that have bboxes
    # write back class = 0, bboxes == [0,0,0,0] to the chips that do not contain building footprints
    #coords,chips,classes = wv.get_labels(args.json_filepath)
    coords,chips= load_building_footprint_nondamaged(args.json_filepath)
    im,box_chip = chip_image_withboxes(arr,coords[chips==image_name],chip_size)
    #im = chip_image(arr,chip_size)

    # debug
    # TODO: if there are black images in test images. Then need to remove black chips here
    # automatic cloud removal if nessessary
    

    
    boxes = []
    scores = []
    classes = []
    k = 0

    images_list = []
    empty_image_idx = []
    for idx, image in enumerate(im):
        # skip chips that do not have buildings, avoid feeding them into inference
        if len(box_chip) == 0 or (len(box_chip) == 1 and np.all(box_chip==0)):
            empty_image_idx.append(idx)
            k = k+1
        else:
            images_list.append(image)
    images = np.array(images_list)
    images.astype(np.uint8)    
    print('number of images without bboxes: ', k)
    print('images shape: ', images.shape)        
    
    i = 0    
    boxes_pred, scores_pred, classes_pred = generate_detections(args.checkpoint,images)
    
    for idx, image in enumerate(im):
        if idx in set(empty_image_idx): 

            box = [[0,0,0,0]]
            score =[[0]]
            clss = [[0]]
            boxes.append(box.astype(np.int32))
            scores.append(score.astype(np.int32))
            classes.append(clss.astype(np.int32))
        else:
            #continue
            box_pred, score_pred, cls_pred = boxes_pred[i], scores_pred[i], classes_pred[i] 
            boxes.append(box_pred)
            scores.append(score_pred)
            classes.append(cls_pred)
            i = i+1
    
    boxes =   np.squeeze(np.array(boxes_pred))
    scores = np.squeeze(np.array(scores_pred))
    classes = np.squeeze(np.array(classes_pred))

    #generate detections
    #boxes, scores, classes = generate_detections(args.checkpoint,images)

    #Process boxes to be full-sized
    width,height,_ = arr.shape
    cwn,chn = (chip_size)
    wn,hn = (int(width/cwn),int(height/chn))


    # changed to 100 in harvey situtation
    #num_preds = 250
    num_preds = 100
    
    # debug
    #bfull = boxes[:wn*hn].reshape((wn,hn,num_preds,4))  # original
    bfull = boxes[:wn*hn].reshape((hn, wn,num_preds,4))

    # debug
    # commented out original transform, because the way the images
    # are chipped are changed
    
    b2 = np.zeros(bfull.shape)
    b2[:,:,:,0] = bfull[:,:,:,1]
    b2[:,:,:,1] = bfull[:,:,:,0]
    b2[:,:,:,2] = bfull[:,:,:,3]
    b2[:,:,:,3] = bfull[:,:,:,2]

    bfull = b2
   


    bfull[:,:,:,0] *= cwn
    bfull[:,:,:,2] *= cwn
    bfull[:,:,:,1] *= chn
    bfull[:,:,:,3] *= chn
    # debug
    #for i in range(wn):
     #   for j in range(hn):
    for i in range(hn):
        for j in range(wn):
            '''
            # original
            bfull[i,j,:,0] += j*cwn
            bfull[i,j,:,2] += j*cwn
            
            bfull[i,j,:,1] += i*chn
            bfull[i,j,:,3] += i*chn
            '''

            
            bfull[i,j,:,0] += i*cwn
            bfull[i,j,:,2] += i*cwn
            
            bfull[i,j,:,1] += j*chn
            bfull[i,j,:,3] += j*chn
            
    bfull = bfull.reshape((hn*wn,num_preds,4))

    
    #only display boxes with confidence > .5
    bs = bfull[scores > .5]
    cs = classes[scores>.5]
    s = args.input.split("/")[::-1]
    draw_bboxes(arr,bs,cs).save("p_bboxes/"+s[0].split(".")[0] + ".png")
    

    with open(args.output,'w') as f:
        for i in range(bfull.shape[0]):
            for j in range(bfull[i].shape[0]):
                #box should be xmin: ymin xmax ymax
                box = bfull[i,j]
                class_prediction = classes[i,j]
                score_prediction = scores[i,j]
                f.write('%d %d %d %d %d %f \n' % \
                    (box[0],box[1],box[2],box[3],int(class_prediction),score_prediction))

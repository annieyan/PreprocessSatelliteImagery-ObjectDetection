# **Disclaimer**

This repository is dual-licensed under Apache 2.0 and Creative Common Attribution-NonCommercial-NoDerivs CC BY-NC-ND 4.0 (see license files attached in this repository).

This repository is modified from the code for the [ the DIUx xView Detection Challenge](https://github.com/DIUx-xView). The paper is available [here](https://arxiv.org/abs/1802.07856). 

This repository is created for [Automatic Damage Annotation on Post-Hurricane Satellite Imagery](https://dds-lab.github.io/disaster-damage-detection/), one of three [projects](https://escience.washington.edu/2018-data-science-for-social-good-projects/) from the 2018 Data Science for Social Good summer fellowship at the University of Washington eScience Institute. 

* Project Lead: Professor Youngjun Choe
* Data Scientist: Valentina Staneva
* Dataset Creation: Sean Chen, Andrew Escay, Chris Haberland, Tessa Schneider, An Yan
* Data processing for training, model training and inference, experiment design: An Yan


# Introduction

Two object detection algorithms, [Single Shot Multibox Detector](https://arxiv.org/abs/1512.02325) and [Faster R-CNN](https://arxiv.org/abs/1506.01497) were applied to satellite imagery for hurricane Harvey provided by [DigitalGlobe Open Data Program](https://www.digitalglobe.com/opendata) and crowd-sourced damaged buildings labels provided by [Tomnod](https://www.tomnod.com/). Our team built dataset for damaged building object detection by placing bounding boxes for damaged buildings whose locations are labelled by Tomnod. For more information about dataset creation, please visit [our website](https://dds-lab.github.io/disaster-damage-detection/data-collecting/). 

We used[tensorflow object detection API](https://github.com/tensorflow/models) to run SSD and Faster R-CNN. We used a baseline model provided by [xView Challenge](https://github.com/DIUx-xView/) as pre-trained model for SSD. This repository contains code performs data processing training, specifically, 1) image chipping, 2) bounding box visualization, 3) train-test split, 4) data augmentation, 5) convert imagery and labels into TFrecord, 6) inference, 7) scoring, 8) cloud removal, 9) black region removal, etc. See [here](https://github.com/annieyan/PreprocessSatelliteImagery-ObjectDetection/wiki/Workflow). We fed the data into SSD and Faster R-CNN and predicted the bounding boxes for damaged and non-damaged buildings. 


# Data Processing for training

The dataset contains 875 satellite imagery of the size 2048 x 2048 in geotiff format. Each of the imagery contains at least one bounding box. The dataset also includes a geojson file containing bounding box location and its class (damaged building or non-damaged building). The dataset contains many invalid bounding boxes that do not cover any building, for example, bounding boxes over the cloud or in the field. Automatic cloud removal method was applied and followed by manual cleaning. Data was then split into training, validation, and test set. 

Imagery were converted to tiff files and chipped into smaller non-overlapping images for training. Different sizes (500 x 500, 300 x 300, and 200 x 200) were experimented and 200 x 200 was chosen because it resulted in better performance. This may because 200 x 200 chips contains less context and buildings appear larger in resultant images than other two sizes. Bounding boxes cross multiple chips were truncated at the edge, among them, those with small width ( < 30 pixels) were discarded. Chips with black region area larger than 10% of its coverage were discarded. Only chips with at least one building were written into TF-record. 

Three training datasets were created with different treatments: 1) Damaged-Only: with only bounding boxes over damaged buildings whose labels came from Tomnod labels; 2) Damaged + non-damaged: bounding boxes of all buildings in the study area were created from building footprints. Buildings without Tomnod labels were treated as another class as non-damaged labels; 3) Augmented damaged and non-damaged: There are about 10 times more non-damaged buildings than damaged buildings. Chips with damaged buildings were then augmented using different random combinations of rotation, flip, zoom-in, Gaussian-blur, Gaussian-noise, change contrast, and change brightness. Additional 200 x 200 chips were created with randomly chosen damaged buildings bounding boxes in the center as a substitute of translation. For details, see [here](https://github.com/annieyan/PreprocessSatelliteImagery-ObjectDetection/wiki/Clean-Data-Run-Statistics-----Harvey).

Then the chips and its associated labels went though optional processing (i.e., data augmentation) and were converted to TFrecord ready for training and testing.    

# Experiments

### SSD 

Inception v2 was chosen as the backbone network for SSD. A single GPU (Tesla K80) machine on AWS was used for model training. Three experiments were conducted using three different training datasets described above. The results are shown in Table 1. 

                            Table 1, Average Precision (AP) Scores for Each Model Run

|                                      | damaged/flooded building | non-damaged building | mAP         |
|--------------------------------------|------------------|----------------------|-------------|
| damaged only                         | 0.0523           | NA                   | 0.0523      |
| damaged + non-damaged                | 0.3352           | 0.5703               | 0.4528      |
| damaged + non-damaged + augmentation | 0.4742           | 0.6223               | 0.5483 |



Training on 2-class produces a far better model than using only damaged building bounding boxes. This may because adding non-damaged buildings data increases model's overall capability to identify buildings, either damaged or non-damaged. Data augmentation helped boosting performance for both damaged and non-damaged buildings. Visualizations of inference results from "Augmented damaged + non-damaged" model are shown in the figures below.


![flooded_large](https://github.com/DDS-Lab/harvey_data_process/blob/master/tomnod_vis/flooded_large.png)
        <center>Predicted damaged/flooded buildings (red) and non-damaged buildings (green)</center>


![](https://github.com/DDS-Lab/harvey_data_process/blob/master/tomnod_vis/blue-tarp.jpg)
![](https://github.com/DDS-Lab/harvey_data_process/blob/master/tomnod_vis/roof-damage.jpg)
![](https://github.com/DDS-Lab/harvey_data_process/blob/master/tomnod_vis/flooded2.jpg)
![](https://github.com/DDS-Lab/harvey_data_process/blob/master/tomnod_vis/flooded3.jpg)

<center>Comparison between ground truth (left panel, '1' denotes damaged/flooded buildings, '2' denotes non-damaged buildings) and predictions (right panel, green denotes damaged buildings and teal denotes non-damaged buildings). The model is able to pick different damaged types including flooded building, blue tarp on the roof, and crashed buildings. </center>


### Faster R-CNN
Faster R-CNN with inception v2 backbone network was applied to the "Augmented damaged + non-damaged" training dataset. Average Precision is 0.31 for damaged/flooded buildings, 0.61 for non-damaged buildings, and 0.46 on average for the two classes. SSD outperformed Faster-RCNN in this case maybe due to the availability of pre-trained model using xView data. 



# Reference:

- Object Detection Baselines in Overhead Imagery with DIUx xView: https://medium.com/@dariusl/object-detection-baselines-in-overhead-imagery-with-diux-xview-c39b1852f24f



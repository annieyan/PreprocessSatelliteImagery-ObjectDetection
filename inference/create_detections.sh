#!/bin/bash

#TESTLIST="20170902_105001000B9D7E00_3030320_jpeg_compressed_08_05.tif 20170831_105001000B95E200_3002321_jpeg_compressed_02_03.tif 20170901_1030010070C13600_3030230_jpeg_compressed_03_09.tif 20170829_1040010032211E00_2110223_jpeg_compressed_03_07.tif 20170829_1040010032211E00_2110200_jpeg_compressed_08_07.tif"
#TESTLIST="20170902_105001000B9D7E00_3030320_jpeg_compressed_08_05.tif"

DIR='/home/ubuntu/anyan/harvey_data/harvey_test_bigtiff_v3/'
for i in "$DIR"/*; do

#for i in $TESTLIST; do
    #echo $i
    echo $(basename "$i")
    python create_detections.py /home/ubuntu/anyan/harvey_data/bboxes_tomnod_2class_noclean.geojson -c /home/ubuntu/anyan/models/research/models/harvey_ssd_inceptionv2_ms_noclean_2class_v2/harvey_ms_2class_v2_96224.pb -cs 200 -o 'preds_output/'$(basename "$i")'.txt' $i
done

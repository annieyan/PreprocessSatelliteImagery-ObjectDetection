'''
Copyright (C) 2018 <eScience Institue at University of Washington>
Licensed under CC BY-NC-ND 4.0 License [see LICENSE-CC BY-NC-ND 4.0.markdown for details] 
Written by An Yan

'''

'''
From training data, create toy validation data for visualization
of model inferences. Note that the toy validation data may contain
training data as well as data used for validation by the model

create one folder: harvey_vis_result_toydata
'''

import argparse
import os
import shutil
import numpy as np
import itertools
from random import shuffle
'''
args:
    abs_dirname: asolute path to all TRAINING chips, i.e., harvey_train_second
    toydir: directory name for toy data
    split_percent: 0.2 for toy data
'''
def seperate_nfiles(abs_dirname, toydir, split_percent):

    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]

    i = 0
    parent_folder = os.path.abspath(abs_dirname + "/../")
    #seperate_subdir = None
    toydir_name = os.path.join(parent_folder, toydir)
    os.mkdir(toydir_name)

    # shuffle files
    #file_count = len(files)
    #perm = np.random.permutation(file_count)
    shuffle(files)
    split_ind = int(split_percent * len(files))
    num_toy = 0

    for idx, f in enumerate(files):
        # create new subdir if necessary
        print('idx', idx)
        if idx < split_ind:
            #subdir_name = os.path.join(abs_dirname, '{0:03d}'.format(i / N + 1))
           # os.mkdir(subdir_name)
           # seperate_subdir = subdir_name

        # copy file to current dir
            f_base = os.path.basename(f)
            shutil.copy(f, os.path.join(toydir_name, f_base))
            num_toy += 1
    print('created toy images: ', num_toy)


def parse_args():
    """Parse command line arguments passed to script invocation."""
    parser = argparse.ArgumentParser(
        description='Split files into multiple subfolders.')
    # src dir is for harvey TRAINING DIR
    parser.add_argument('src_dir', help='source directory')

    return parser.parse_args()


def main():
    """Module's main entry point (zopectl.command)."""
    args = parse_args()
    src_dir = args.src_dir

    if not os.path.exists(src_dir):
        raise Exception('Directory does not exist ({0}).'.format(src_dir))

    #move_files(os.path.abspath(src_dir))
    toy_dir = 'harvey_vis_result_toydata'
    
    seperate_nfiles(os.path.abspath(src_dir),toy_dir, 0.2)


if __name__ == '__main__':
    main()
                                                                                    

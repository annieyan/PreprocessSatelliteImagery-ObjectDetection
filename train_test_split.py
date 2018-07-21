'''
Split train : test = 7:3
create two folders: harvey_train_(round_num) and harvey_test_(round_num)
create two geojson files respectively for train and test
'''

import argparse
import os
import shutil
import numpy as np
import itertools
from random import shuffle
'''
args:
    abs_dirname: asolute path to all chips
    traindir: directory name for training data. ie, harvey_train_first
    testdir: directory name for test data.
    split_percent: 0.8 for 80% training data, 20% test data
'''
def seperate_nfiles(abs_dirname, traindir, testdir, split_percent):

    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]

    i = 0
    parent_folder = os.path.abspath(abs_dirname + "/../")
    #seperate_subdir = None
    traindir_name = os.path.join(parent_folder, traindir)
    testdir_name = os.path.join(parent_folder, testdir)
    #seperate_subdir = subdir_name
    os.mkdir(traindir_name)
    os.mkdir(testdir_name)
    print('traindir_name', traindir_name)

    # shuffle files
    #file_count = len(files)
    #perm = np.random.permutation(file_count)
    shuffle(files)
    split_ind = int(split_percent * len(files))
    num_train = 0
    num_test = 0

    for idx, f in enumerate(files):
        # create new subdir if necessary
        print('idx', idx)
        if idx < split_ind:
            #subdir_name = os.path.join(abs_dirname, '{0:03d}'.format(i / N + 1))
           # os.mkdir(subdir_name)
           # seperate_subdir = subdir_name

        # copy file to current dir
            f_base = os.path.basename(f)
            shutil.copy(f, os.path.join(traindir_name, f_base))
            num_train += 1
        else:
            # go to test dir
            f_base = os.path.basename(f)
            shutil.copy(f, os.path.join(testdir_name, f_base))
            num_test +=1
    print('created training images: ', num_train)
    print('created test images: ', num_test)


def parse_args():
    """Parse command line arguments passed to script invocation."""
    parser = argparse.ArgumentParser(
        description='Split files into multiple subfolders.')

    parser.add_argument('src_dir', help='source directory')

    return parser.parse_args()


def main():
    """Module's main entry point (zopectl.command)."""
    args = parse_args()
    src_dir = args.src_dir

    if not os.path.exists(src_dir):
        raise Exception('Directory does not exist ({0}).'.format(src_dir))

    #move_files(os.path.abspath(src_dir))
    train_dir = 'harvey_train_first'
    test_dir = 'harvey_test_first'
    seperate_nfiles(os.path.abspath(src_dir), train_dir, test_dir, 0.7)


if __name__ == '__main__':
    main()
                                                                                    

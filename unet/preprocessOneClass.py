import os
import warnings

import numpy as np

from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.transform import resize

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


class OneClassPreprocessor:
    def __init__(self, width, height, channels, train_path_in, test_path_in, train_path_out, test_path_out ):
        self.IMG_WIDTH = width
        self.IMG_HEIGHT = height
        self.IMG_CHANNELS = channels
        self.TRAIN_PATH = train_path_in
        self.TEST_PATH = test_path_in

        self.PREPROCESSED_TRAIN_PATH = train_path_out
        self.PREPROCESSED_TEST_PATH = test_path_out
    
    def preprocess( self ):
        # Get train and test IDs
        # Returns a list of folder names in the training and testing paths
        train_ids = next(os.walk(self.TRAIN_PATH))[1]
        test_ids = next(os.walk(self.TEST_PATH))[1]
        
        print('Getting and resizing training images ... ')
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            path = self.TRAIN_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:,:,:self.IMG_CHANNELS]
            img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
            img = img.astype(np.uint8)
            imsave( self.PREPROCESSED_TRAIN_PATH + "images/%04d.png" % n, img )
            
            mask = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 1), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
                mask = np.maximum(mask, mask_)
            mask = mask.astype(np.uint8)
            imsave( self.PREPROCESSED_TRAIN_PATH + "masks/%04d.png" % n, mask )

        # Get and resize test images
        test_sizes_string = ""
        print('Getting and resizing test images ... ')

        for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
            path = self.TEST_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:,:,:self.IMG_CHANNELS]
            
            test_sizes_string += "%d %d\n" % ( img.shape[0], img.shape[1] );
            
            img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
            img = img.astype(np.uint8)
            imsave( self.PREPROCESSED_TEST_PATH + "images/%04d.png" % n, img )
        
        file_sizes = open( self.PREPROCESSED_TEST_PATH + "sizes.txt", "w" )
        file_sizes.write( test_sizes_string )
        file_sizes.close()
        
        print('Done!')
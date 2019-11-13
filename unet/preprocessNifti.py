import os
import warnings

import numpy as np
import nibabel as nib
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
        train_data = next(os.walk(self.TRAIN_PATH))[1]
        train_images_path= train_data[0]
        train_labels_path= train_data[1]
        train_image_ids= next(os.walk(self.TRAIN_PATH + train_images_path + '/'))[2]
        train_labels_ids= next(os.walk(self.TRAIN_PATH + train_labels_path + '/'))[2]
        print('Getting and resizing training images and labels... ')
        for n, id_ in tqdm(enumerate(train_image_ids), total=len(train_image_ids)):
            proxy_img = nib.load( self.TRAIN_PATH + train_images_path + '/' + id_)
            canonical_img = nib.as_closest_canonical(proxy_img)
            
            self._base_image_data = canonical_img.get_fdata()
            
            print( self._base_image_data.shape)
           
            numslices0 = self._base_image_data.shape[ 0 ]
            numslices1 = self._base_image_data.shape[ 1 ]
            numslices2 = self._base_image_data.shape[ 2 ]
           
            name = id_.split('.')[0]

            for i in range(1,numslices0):
                img = self._base_image_data[i,:,:]
                img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
                imsave( self.PREPROCESSED_TRAIN_PATH + "%s-%04d-image.png" % (name, i), img )

            for j in range(1,numslices1):
                img = self._base_image_data[:,j,:]
                img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
                imsave( self.PREPROCESSED_TRAIN_PATH + "%s-%04d-image.png" % (name, i + j), img )

            for k in range(1,numslices2):
                img = self._base_image_data[:,:,k]
                img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
                imsave( self.PREPROCESSED_TRAIN_PATH + "%s-%04d-image.png" % (name, j + k), img )

        for n, id_ in tqdm(enumerate(train_labels_ids), total=len(train_labels_ids)):
            proxy_img = nib.load( self.TRAIN_PATH + train_labels_path + '/' + id_)
            canonical_img = nib.as_closest_canonical(proxy_img)
            
            self._base_image_data = canonical_img.get_fdata()
            
            print( self._base_image_data.shape)
           
            numslices0 = self._base_image_data.shape[ 0 ]
            numslices1 = self._base_image_data.shape[ 1 ]
            numslices2 = self._base_image_data.shape[ 2 ]
           
            name = id_.split('.')[0].split('-')[0]

            for i in range(1,numslices0):
                img = self._base_image_data[i,:,:]
                img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
                imsave( self.PREPROCESSED_TRAIN_PATH + "%s-%04d-label.png" % (name, i), img )

            for j in range(1,numslices1):
                img = self._base_image_data[:,j,:]
                img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
                imsave( self.PREPROCESSED_TRAIN_PATH + "%s-%04d-label.png" % (name, i + j), img )

            for k in range(1,numslices2):
                img = self._base_image_data[:,:,k]
                img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
                imsave( self.PREPROCESSED_TRAIN_PATH + "%s-%04d-label.png" % (name, j + k), img )  
     
           


        
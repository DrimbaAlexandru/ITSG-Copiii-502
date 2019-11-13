import os
import warnings

import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage.io import  imsave
from skimage.transform import resize
from skimage.color import gray2rgb

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


class NIfTIPreprocessor:
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
        # Returns a list of image file names in the training and testing paths
        # Label files are expected to be image_name -label
        train_image_ids = next(os.walk(self.TRAIN_PATH + 'images/'))[2]
        train_labels_ids = next(os.walk(self.TRAIN_PATH + 'labels/'))[2]

        print('Getting and resizing training images and labels... ')
        for n, id_ in enumerate(train_image_ids):
            # Process images
            print( "Exporting image %d/%d" % ( n, len(train_image_ids) ) )

            proxy_img = nib.load( self.TRAIN_PATH + 'images/' + id_)
            canonical_img = nib.as_closest_canonical(proxy_img)
            
            image_data = canonical_img.get_fdata()
            min_max = ( image_data.min(), image_data.max() )
            print( image_data.shape )
           
            numslices0 = image_data.shape[ 0 ]
            numslices1 = image_data.shape[ 1 ]
            numslices2 = image_data.shape[ 2 ]
           
            name = id_.split('.')[0]

            for i in tqdm( range(numslices0) ):
                img = image_data[i,:,:]
                img = np.flip( img.T, axis = 0 )
                if len( img.shape ) == 2 or ( self.IMG_CHANNELS == 3 and img.shape[2] != self.IMG_CHANNELS ):
                    img = gray2rgb( img )
                img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True )
                img = img * ( 255 / min_max[1] )
                img = img.astype(np.uint8)
                imsave( self.PREPROCESSED_TRAIN_PATH + "images/AXIS0-%s-%04d.png" % (name, i), img )

            for j in tqdm(range(numslices1)):
                img = image_data[:,j,:]
                img = np.flip( img.T, axis = 0 )
                if len( img.shape ) == 2 or ( self.IMG_CHANNELS == 3 and img.shape[2] != self.IMG_CHANNELS ):
                    img = gray2rgb( img )
                img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
                img = img * ( 255 / min_max[1] )
                img = img.astype(np.uint8)
                imsave( self.PREPROCESSED_TRAIN_PATH + "images/AXIS1-%s-%04d.png" % (name, j), img )

            for k in tqdm(range(numslices2)):
                img = image_data[:,:,k]
                img = np.flip( img.T, axis = 0 )
                if len( img.shape ) == 2 or ( self.IMG_CHANNELS == 3 and img.shape[2] != self.IMG_CHANNELS ):
                    img = gray2rgb( img )
                img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
                img = img * ( 255 / min_max[1] )
                img = img.astype(np.uint8)
                imsave( self.PREPROCESSED_TRAIN_PATH + "images/AXIS2-%s-%04d.png" % (name, k), img )

            # Process labels
            proxy_img = nib.load( self.TRAIN_PATH + 'labels/' + name + "-label.nii.gz")
            canonical_img = nib.as_closest_canonical(proxy_img)
            
            image_data = canonical_img.get_fdata()
            min_max = ( image_data.min(), image_data.max() )

            print( image_data.shape )
           
            numslices0 = image_data.shape[ 0 ]
            numslices1 = image_data.shape[ 1 ]
            numslices2 = image_data.shape[ 2 ]
           
            name = id_.split('.')[0].split('-')[0]

            for i in tqdm(range(numslices0)):
                img = image_data[i,:,:]
                img = np.flip( img.T, axis = 0 )
                if len( img.shape ) == 2 or ( self.IMG_CHANNELS == 3 and img.shape[2] != self.IMG_CHANNELS ):
                    img = gray2rgb( img )
                img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True, order = 0, anti_aliasing = False)
                img = img * ( 255 / min_max[1] )
                img = img.astype(np.uint8)
                imsave( self.PREPROCESSED_TRAIN_PATH + "masks/AXIS0-%s-%04d.png" % (name, i), img )

            for j in tqdm(range(1,numslices1)):
                img = image_data[:,j,:]
                img = np.flip( img.T, axis = 0 )
                if len( img.shape ) == 2 or ( self.IMG_CHANNELS == 3 and img.shape[2] != self.IMG_CHANNELS ):
                    img = gray2rgb( img )
                img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True, order = 0, anti_aliasing = False)
                img = img * ( 255 / min_max[1] )
                img = img.astype(np.uint8)
                imsave( self.PREPROCESSED_TRAIN_PATH + "masks/AXIS1-%s-%04d.png" % (name, j), img )

            for k in tqdm(range(1,numslices2)):
                img = image_data[:,:,k]
                img = np.flip( img.T, axis = 0 )
                if len( img.shape ) == 2 or ( self.IMG_CHANNELS == 3 and img.shape[2] != self.IMG_CHANNELS ):
                    img = gray2rgb( img )
                img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True, order = 0, anti_aliasing = False)
                img = img * ( 255 / min_max[1] )
                img = img.astype(np.uint8)
                imsave( self.PREPROCESSED_TRAIN_PATH + "masks/AXIS2-%s-%04d.png" % (name, k), img )

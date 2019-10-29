import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imsave, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score, up_opt = tf.compat.v1.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

class Unet_model:
    def __init__(self, width, height, channels, train_path_in, test_path_in, train_path_out, test_path_out ):
        self.IMG_WIDTH = width
        self.IMG_HEIGHT = height
        self.IMG_CHANNELS = channels
        self.TRAIN_PATH = train_path_in
        self.TEST_PATH = test_path_in

        self.PREPROCESSED_TRAIN_PATH = train_path_out
        self.PREPROCESSED_TEST_PATH = test_path_out
        
        self.MODEL_PATH = "my_model.h5"
        
        
    def load_images( self ):
        print( "Read test and traing images from the disk" )
    
        # Get train and test IDs
        # Returns a list of file names in the training and testing paths
        train_ids = next( os.walk( self.PREPROCESSED_TRAIN_PATH + "images/" ) )[2]
        test_ids = next( os.walk( self.PREPROCESSED_TEST_PATH + "images/" ) )[2]

        # Get and resize train images and masks
        self.train_images = np.zeros((len(train_ids), self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
        self.train_masks  = np.zeros((len(train_ids), self.IMG_HEIGHT, self.IMG_WIDTH, 1), dtype=np.bool)
        self.test_images  = np.zeros((len(test_ids),  self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)

        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            img = imread( self.PREPROCESSED_TRAIN_PATH + "images/" + id_ )[:,:,:self.IMG_CHANNELS]   
            self.train_images[n] = img
            mask = imread( self.PREPROCESSED_TRAIN_PATH + "masks/" + id_ )   
            mask = np.expand_dims( mask, axis=-1)
            self.train_masks[n] = mask
            
        for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
            img = imread( self.PREPROCESSED_TEST_PATH + "images/" + id_ )[:,:,:self.IMG_CHANNELS]   
            self.test_images[n] = img
            
        self.sizes_test = []
        file_sizes = open( self.PREPROCESSED_TEST_PATH + "sizes.txt", "r" )
        for line in file_sizes.readlines():
            self.sizes_test.append( ( int( line.split(' ')[0] ), int( line.split(' ')[1] ) ) )    
        file_sizes.close()

        # Check if training data looks all right
        ix = random.randint(0, len(train_ids))
        imshow(self.train_images[ix])
        plt.show()
        imshow(np.squeeze(self.train_masks[ix]))
        plt.show()
    
        
    def create_model( self ):
        print( "Build U-Net model" )
            
        # Build U-Net model
        inputs = Input( ( self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS ) )
        s = Lambda(lambda x: x / 255) (inputs)

        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
        c1 = Dropout(0.1) (c1)
        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
        c2 = Dropout(0.1) (c2)
        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
        c3 = Dropout(0.2) (c3)
        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
        c4 = Dropout(0.2) (c4)
        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
        c5 = Dropout(0.3) (c5)
        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
        c6 = Dropout(0.2) (c6)
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
        c7 = Dropout(0.2) (c7)
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
        c8 = Dropout(0.1) (c8)
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
        c9 = Dropout(0.1) (c9)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

        self.model = Model(inputs=[inputs], outputs=[outputs])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
        self.model.summary()
        
        
    def save_model( self ):
        # Save the model
        self.model.save( self.MODEL_PATH )

        
    def load_model( self ):
        self.model = load_model( self.MODEL_PATH, custom_objects={'mean_iou': mean_iou} )
        
        
    def fit_model( self, epochs = 50 ):
        print( "Fit model" )
        
        # Fit model
        earlystopper = EarlyStopping( patience=5, verbose=1 )
        checkpointer = ModelCheckpoint( self.MODEL_PATH, verbose=1, save_best_only=True )
        results = self.model.fit( self.train_images, self.train_masks, validation_split=0.1, batch_size=16, epochs=epochs, 
                                  callbacks=[earlystopper, checkpointer] )


    def predict_from_model( self ):
        print( "Predict on train, val and test")
        input()

        # Predict on train, val and test
        self.load_model()
        preds_train = self.model.predict(self.train_images[:int(self.train_images.shape[0]*0.9)], verbose=1)
        preds_val = self.model.predict(self.train_images[int(self.train_images.shape[0]*0.9):], verbose=1)
        preds_test = self.model.predict(self.test_images, verbose=1)

        # Threshold predictions
        preds_train_t = (preds_train > 0.5).astype(np.uint8)
        preds_val_t = (preds_val > 0.5).astype(np.uint8)
        preds_test_t = (preds_test > 0.5).astype(np.uint8)

        # Create list of upsampled test masks
        preds_test_upsampled = []
        for i in range(len(preds_test)):
            img = np.squeeze(preds_test[i])
            imsave( self.PREPROCESSED_TEST_PATH + "generated_masks/%04d.png" % i, img )
            preds_test_upsampled.append(resize( img, 
                                               (self.sizes_test[i][0], self.sizes_test[i][1]), 
                                                mode='constant', preserve_range=True))


        print( "Perform a sanity check on some random training samples")
        # Perform a sanity check on some random training samples
        ix = random.randint(0, len(preds_train_t))
        imshow(self.train_images[ix])
        plt.show()
        imshow(np.squeeze(self.train_masks[ix]))
        plt.show()
        imshow(np.squeeze(preds_train_t[ix]))
        plt.show()

        
        print( "Perform a sanity check on some random validation samples")
        # Perform a sanity check on some random validation samples
        ix = random.randint(0, len(preds_val_t))
        imshow(self.train_images[int(self.train_images.shape[0]*0.9):][ix])
        plt.show()
        imshow(np.squeeze(self.train_masks[int(self.train_masks.shape[0]*0.9):][ix]))
        plt.show()
        imshow(np.squeeze(preds_val_t[ix]))
        plt.show()
        
        
        print( "Perform a sanity check on some random testing samples")
        # Perform a sanity check on some random testing samples
        ix = random.randint(0, len(preds_test_t))
        imshow(self.test_images[ix])
        plt.show()
        imshow(np.squeeze(preds_test_upsampled[ix]))
        plt.show()
import os
import sys
import random
import warnings

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.io import imread, imsave, imshow
from skimage.transform import resize

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# m * x * y * n
# m - number of images
# x, y - image dimensions
# n - number of classes
def iou_coef_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return 1 - iou
  
def dice_coef_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return 1 - dice


class Unet_model:
    def __init__( self,
                  width,
                  height,
                  channels,
                  train_path_in,
                  test_path_in,
                  train_path_out,
                  test_path_out,
                  classes = [ ( ( 0, 0, 0 ), "Background" ), ( ( 255, 255, 255 ), "Class 1" ) ] ):
                 
        self.IMG_WIDTH = width
        self.IMG_HEIGHT = height
        self.IMG_CHANNELS = channels
        self.TRAIN_PATH = train_path_in
        self.TEST_PATH = test_path_in
        self.CLASSES = classes
        self.NUM_CLASSES = len( classes )

        self.PREPROCESSED_TRAIN_PATH = train_path_out
        self.PREPROCESSED_TEST_PATH = test_path_out
        
        self.MODEL_PATH = "my_model.h5"

        self.VALIDATION_SPLIT = 0.1

    def load_images( self ):
        print( "Read test and traing images from the disk" )
    
        # Get train and test IDs
        # Returns a list of file names in the training and testing paths
        train_ids = next( os.walk( self.PREPROCESSED_TRAIN_PATH + "images/" ) )[2]
        test_ids = next( os.walk( self.PREPROCESSED_TEST_PATH + "images/" ) )[2]

        # Get and resize train images and masks
        self.train_images = np.zeros( ( len( train_ids ), self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS ), dtype=np.uint8)
        self.train_masks  = np.zeros( ( len( train_ids ), self.IMG_HEIGHT, self.IMG_WIDTH, self.NUM_CLASSES ), dtype=np.bool)
        self.test_images  = np.zeros( ( len( test_ids ),  self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS ), dtype=np.uint8)

        ones = np.ones( ( self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS ), dtype=np.uint8 )
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            img = imread( self.PREPROCESSED_TRAIN_PATH + "images/" + id_ )[:,:,:self.IMG_CHANNELS]
            if len( img.shape ) == 2 or img.shape[2] != self.IMG_CHANNELS:
                raise BaseException("Invalid channel number in training image " + id_ )
            self.train_images[n] = img

            mask = imread( self.PREPROCESSED_TRAIN_PATH + "masks/" + id_ )
            if len( mask.shape ) == 2 or mask.shape[2] != self.IMG_CHANNELS:
                raise BaseException("Invalid channel number in mask image " + id_ )

            classes = np.zeros( ( self.NUM_CLASSES, self.IMG_HEIGHT, self.IMG_WIDTH ), dtype=np.uint8 )
            for i, ( color, _ ) in enumerate( self.CLASSES ):
                curr_class_mask = np.all( np.equal( mask, color * ones ), axis = -1 )
                # plt.show()
                # imshow( curr_class_mask )
                classes[ i ] = curr_class_mask
            # Swap classes' axes from ( n, x, y ) to ( x, y, n )
            classes = np.swapaxes( classes, 0, 2 )
            classes = np.swapaxes( classes, 0, 1 )
            self.train_masks[ n ] = classes

        print( self.train_masks[ 0 ].shape )
            
        for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
            img = imread( self.PREPROCESSED_TEST_PATH + "images/" + id_ )[:,:,:self.IMG_CHANNELS]   
            self.test_images[n] = img
            
        self.sizes_test = []
        file_sizes = open( self.PREPROCESSED_TEST_PATH + "sizes.txt", "r" )
        for line in file_sizes.readlines():
            self.sizes_test.append( ( int( line.split(' ')[0] ), int( line.split(' ')[1] ) ) )    
        file_sizes.close()

    def save_model( self ):
        # Save the model
        self.model.save( self.MODEL_PATH )

    def load_model( self ):
        self.model = load_model( self.MODEL_PATH, custom_objects={'iou_coef_loss': iou_coef_loss,
                                                                  'dice_coef_loss': dice_coef_loss,
                                                                  'metrics': [ iou_coef_loss, dice_coef_loss, "accuracy" ]} )

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

        outputs = Conv2D(self.NUM_CLASSES, (1, 1), activation='softmax') (c9)

        self.model = Model(inputs=[inputs], outputs=[outputs])
        #self.model.compile(optimizer='adam', loss=iou_coef_loss, metrics=[ iou_coef_loss, dice_coef_loss, "accuracy" ])
        self.model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[ iou_coef_loss, dice_coef_loss, "accuracy" ] )
        self.model.summary()
        
    def fit_model( self, epochs = 50 ):
        print( "Fit model" )
        
        # Fit model
        earlystopper = EarlyStopping( patience=5, verbose=1 )
        checkpointer = ModelCheckpoint( self.MODEL_PATH, verbose=1, save_best_only=True, monitor="val_iou_coef_loss" )
        #checkpointer = ModelCheckpoint( self.MODEL_PATH, verbose=1, save_best_only=True )
        results = self.model.fit( self.train_images, self.train_masks, validation_split=self.VALIDATION_SPLIT, batch_size=16, epochs=epochs,
                                  callbacks=[earlystopper, checkpointer] )

    # prediction has the shape x y m
    def _prediction_to_mask( self, prediction ):
        mask = np.zeros( ( self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS ) )
        idxs = np.argmax( prediction, axis=-1 )
        for cn, ( k, _ ) in enumerate( self.CLASSES ):
            mask[ idxs == cn ] = k
        return mask


    # predictions has the shape n x y m
    def _predictions_to_mask( self, predictions ):
        masks = np.zeros( predictions.shape[:3] + (self.IMG_CHANNELS,), dtype=np.uint8 )
        for i, img in enumerate( predictions ):
            masks[ i ] = self._prediction_to_mask( img )
        return masks

    def predict_from_model( self ):
        print( "Predict on train, val and test")

        # Predict on train, val and test
        preds_val = self.model.predict(self.train_images[int(self.train_images.shape[0] * self.VALIDATION_SPLIT ):], verbose=1)
        preds_train = self.model.predict(self.train_images[:int(self.train_images.shape[0] * self.VALIDATION_SPLIT )], verbose=1)
        preds_test = self.model.predict(self.test_images, verbose=1)

        len_train = len(preds_train)
        len_val = len(preds_val)
        len_test = len(preds_test)

        preds_train_t = self._predictions_to_mask(preds_train)
        preds_val_t = self._predictions_to_mask(preds_val)
        preds_test_t = self._predictions_to_mask(preds_test)

        # Create list of upsampled test masks
        for i in range( len_test ):
            img = preds_test_t[ i ]
            imsave( self.PREPROCESSED_TEST_PATH + "generated_masks/%04d.png" % i, img )
            #resize( img, ( self.sizes_test[i][0], self.sizes_test[i][1]), mode='constant', preserve_range=True )

        print( "Perform a sanity check on some random training samples")
        # Perform a sanity check on some random training samples
        ix = random.randint(0, len_train - 1 )
        imshow( self.train_images[ ix ] )
        plt.show()
        imshow( self._prediction_to_mask( self.train_masks[ ix ] ) )
        plt.show()
        imshow( preds_train_t[ ix ] )
        plt.show()

        print( "Perform a sanity check on some random validation samples")
        # Perform a sanity check on some random validation samples
        ix = random.randint(0, len_val - 1 )
        imshow( self.train_images[ len_train + ix ] )
        plt.show()
        imshow( self._prediction_to_mask( self.train_masks[ len_train + ix ]) )
        plt.show()
        imshow( preds_val_t[ ix ] )
        plt.show()

        print( "Perform a sanity check on some random testing samples")
        # Perform a sanity check on some random testing samples
        ix = random.randint(0, len_test - 1 )
        imshow( self.test_images[ ix ] )
        plt.show()
        imshow( preds_test_t[ ix ] )
        plt.show()
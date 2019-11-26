import os
import random
import warnings
import datetime

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
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# m * x * y * n
# m - number of images
# x, y - image dimensions
# n - number of classes
def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou
  
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def iou_coef_loss(yt,yp,smooth=1):
    return 1 - iou_coef(yt,yp,smooth)

def dice_coef_loss(yt,yp,smooth=1):
    return 1 - dice_coef(yt,yp,smooth)

class Unet_model:
    def __init__( self,
                  width,
                  height,
                  channels,
                  train_path_in,
                  test_path_in,
                  train_path_out,
                  test_path_out,
                  test_data_labeled,
                  classes = [ ( ( 0, 0, 0 ), "Background" ), ( ( 255, 255, 255 ), "Class 1" ) ] ):
                 
        self.IMG_WIDTH = width
        self.IMG_HEIGHT = height
        self.IMG_CHANNELS = channels
        self.TRAIN_PATH = train_path_in
        self.TEST_PATH = test_path_in
        self.CLASSES = classes
        self.NUM_CLASSES = len( classes )
        self.IS_TEST_DATA_LABELED = test_data_labeled

        self.PREPROCESSED_TRAIN_PATH = train_path_out
        self.PREPROCESSED_TEST_PATH = test_path_out
        
        self.MODEL_PATH = "my_model.h5"
        self.LOG_DIR = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(self.LOG_DIR)

        self.VALIDATION_SPLIT = 0.1

        self.ones = np.ones( ( self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS ), dtype=np.uint8 )

    def mask_to_classes( self, mask ):
        classes = np.zeros( ( self.NUM_CLASSES, self.IMG_HEIGHT, self.IMG_WIDTH ), dtype=np.bool )
        for i, ( color, _ ) in enumerate( self.CLASSES ):
            curr_class_mask = np.all( np.equal( mask, color * self.ones ), axis = -1 )
            # plt.show()
            # imshow( curr_class_mask )
            classes[ i ] = curr_class_mask
        # Swap classes' axes from ( n, x, y ) to ( x, y, n )
        classes = np.swapaxes( classes, 0, 2 )
        classes = np.swapaxes( classes, 0, 1 )
        return classes

    def masks_to_classes(self, masks):
        classes_masks = np.zeros( ( len( masks ), self.IMG_HEIGHT, self.IMG_WIDTH, self.NUM_CLASSES ), dtype=np.bool)
        for i, mask in tqdm(enumerate(masks), total=len(masks)):
            classes = self.mask_to_classes(mask)
            classes_masks[ i ] = classes
        return classes_masks

    def load_training_images( self ):
        print( "Read training images from the disk" )
    
        # Get train IDs
        # Returns a list of file names in the training path
        train_ids = next( os.walk( self.PREPROCESSED_TRAIN_PATH + "images/" ) )[2]

        # Get and resize train images and masks
        self.train_images = np.zeros( ( len( train_ids ), self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS ), dtype=np.uint8)
        self.train_masks  = np.zeros( ( len( train_ids ), self.IMG_HEIGHT, self.IMG_WIDTH, self.NUM_CLASSES ), dtype=np.bool)

        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            img = imread( self.PREPROCESSED_TRAIN_PATH + "images/" + id_ )[:,:,:self.IMG_CHANNELS]
            if len( img.shape ) == 2 or img.shape[2] != self.IMG_CHANNELS:
                raise BaseException("Invalid channel number in training image " + id_ )
            self.train_images[n] = img

            mask = imread( self.PREPROCESSED_TRAIN_PATH + "masks/" + id_ )
            if len( mask.shape ) == 2 or mask.shape[2] != self.IMG_CHANNELS:
                raise BaseException("Invalid channel number in mask image " + id_ )
            self.train_masks[ n ] = self.mask_to_classes(mask)

    def load_testing_images( self ):
        print( "Read testing images from the disk" )

        # Get test IDs
        # Returns a list of file names in the testing path
        test_ids = next( os.walk( self.PREPROCESSED_TEST_PATH + "images/" ) )[2]

        # Get and resize train images and masks
        self.test_images = np.zeros( ( len( test_ids ), self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS ), dtype=np.uint8)
        self.test_masks  = np.zeros( ( len( test_ids ), self.IMG_HEIGHT, self.IMG_WIDTH, self.NUM_CLASSES ), dtype=np.bool)

        for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
            img = imread( self.PREPROCESSED_TEST_PATH + "images/" + id_ )[:,:,:self.IMG_CHANNELS]
            if len( img.shape ) == 2 or img.shape[2] != self.IMG_CHANNELS:
                raise BaseException("Invalid channel number in test image " + id_ )
            self.test_images[n] = img

            if self.IS_TEST_DATA_LABELED:
                mask = imread( self.PREPROCESSED_TEST_PATH + "masks/" + id_ )
                if len( mask.shape ) == 2 or mask.shape[2] != self.IMG_CHANNELS:
                    raise BaseException("Invalid channel number in mask image " + id_ )
                self.test_masks[ n ] = self.mask_to_classes(mask)

    def save_model( self ):
        # Save the model
        self.model.save( self.MODEL_PATH )


    def load_model( self ):
        self.model = load_model( self.MODEL_PATH, custom_objects={'iou_coef_loss': iou_coef_loss,
                                                                  'dice_coef_loss': dice_coef_loss,
                                                                  'iou_coef': iou_coef,
                                                                  'dice_coef': dice_coef,
                                                                  'metrics': [ iou_coef, dice_coef, iou_coef_loss, dice_coef_loss, "accuracy" ]} )

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
        self.model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[ iou_coef, dice_coef, iou_coef_loss, dice_coef_loss, "accuracy" ] )
        self.model.summary()
        
    def fit_model( self, epochs = 50 ):
        print( "Fit model" )
        
        # Fit model
        tensorboard = TensorBoard(log_dir=self.LOG_DIR, histogram_freq=1)
        earlystopper = EarlyStopping(patience=5, verbose=1)
        checkpointer = ModelCheckpoint(self.MODEL_PATH, verbose=1, save_best_only=True, monitor="val_iou_coef_loss")
        # checkpointer = ModelCheckpoint( self.MODEL_PATH, verbose=1, save_best_only=True )

        self.model.fit(self.train_images, self.train_masks, validation_split=self.VALIDATION_SPLIT,
                       batch_size=16, epochs=epochs,
                       callbacks=[earlystopper, checkpointer, tensorboard])

    # prediction has the shape x y m
    def _prediction_to_mask( self, prediction ):
        mask = np.zeros( ( self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS ), dtype=np.uint8 )
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

    def predict_one(self, path_in, path_out=None ):
        input_images  = np.zeros( ( 1,  self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS ), dtype=np.uint8)
        input_images[0] = imread( path_in )[:,:,:self.IMG_CHANNELS]
        preds = self.model.predict(input_images, verbose=0)
        preds_mask = self._predictions_to_mask(preds)
        if path_out is not None:
            imsave(path_out, preds_mask[0])

        return preds_mask[0]

    def predict_all(self, base_folder, write_output=True):
        # Get IDs
        # Returns a list of file names in the given path
        ids = next( os.walk( base_folder + "images/" ) )[2]

        pred_masks = np.zeros( ( len( ids ), self.IMG_HEIGHT, self.IMG_WIDTH, self.NUM_CLASSES ), dtype=np.uint8)

        if write_output:
            os.makedirs(base_folder + "masks_generated/",exist_ok=True)

        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            pred_masks[n] = self.predict_one(base_folder + "images/" + id_,
                                             base_folder + "masks_generated/" + id_ if write_output else None )
        return pred_masks

    def predict_from_model( self ):
        self.predict_all(self.PREPROCESSED_TEST_PATH,True)

    def evaluate_model(self):

        input_train = self.train_images[int(self.train_images.shape[0] * self.VALIDATION_SPLIT ):]
        input_val   = self.train_images[:int(self.train_images.shape[0] * self.VALIDATION_SPLIT )]

        masks_train = self.train_masks[int(self.train_images.shape[0] * self.VALIDATION_SPLIT ):]
        masks_val  = self.train_masks[:int(self.train_images.shape[0] * self.VALIDATION_SPLIT )]

        metrics_train = self.model.evaluate(input_train,masks_train,batch_size=16)
        metrics_val = self.model.evaluate(input_val,masks_val,batch_size=16)

        if( self.IS_TEST_DATA_LABELED ):
            metrics_test = self.model.evaluate(self.test_images,self.test_masks,batch_size=16)

        resultsFile = open(self.LOG_DIR + "\\results.txt", "w")

        resultsFile.write("Number of training samples: " + str( len(input_train) ) )
        resultsFile.write("\r\nNumber of validation samples: " + str( len(input_val) ) )
        if( self.IS_TEST_DATA_LABELED ):
            resultsFile.write("\r\nNumber of testing samples: " + str( len(self.test_images) ) )

        resultsFile.write("\r\nTraining results:" )
        resultsFile.write("\r\n  IoU:  " + str( metrics_train[1] ) )
        resultsFile.write("\r\n  Dice: " + str( metrics_train[2] ) )
        resultsFile.write("\r\n")

        resultsFile.write("\r\nValidation results:" )
        resultsFile.write("\r\n  IoU:  " + str( metrics_val[1] ) )
        resultsFile.write("\r\n  Dice: " + str( metrics_val[2] ) )
        resultsFile.write("\r\n")

        if( self.IS_TEST_DATA_LABELED ):
            resultsFile.write("\r\nTest results:" )
            resultsFile.write("\r\n  IoU:  " + str( metrics_test[1] ) )
            resultsFile.write("\r\n  Dice: " + str( metrics_test[2] ) )
            resultsFile.write("\r\n")

        resultsFile.close()

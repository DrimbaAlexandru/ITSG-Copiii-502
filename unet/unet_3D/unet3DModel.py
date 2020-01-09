import os
import random
import warnings
import datetime

import numpy as np

from tqdm import tqdm
from skimage.transform import resize, rescale

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling3D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K

import utils.utils as myUtils

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# m * x * y * z* n
# m - number of images
# x, y, z - image dimensions
# n - number of classes
def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3,4])
    union = K.sum(y_true,[1,2,3,4])+K.sum(y_pred,[1,2,3,4])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou
  
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3,4])
    union = K.sum(y_true, axis=[1,2,3,4]) + K.sum(y_pred, axis=[1,2,3,4])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def iou_coef_loss(yt,yp,smooth=1):
    return 1 - iou_coef(yt,yp,smooth)

def dice_coef_loss(yt,yp,smooth=1):
    return 1 - dice_coef(yt,yp,smooth)

class Unet3DModel:
    def __init__( self,
                  classes,
                  size,
                  train_path_in = None,
                  test_path_in = None,
                  train_path_out = None,
                  test_path_out = None,
                  test_data_labeled = None,
                  ):
                 
        self.IMG_SIZE = size
        self.TRAIN_PATH = train_path_in
        self.TEST_PATH = test_path_in
        self.CLASSES = classes
        self.NUM_CLASSES = len( classes )
        self.IS_TEST_DATA_LABELED = test_data_labeled

        self.PREPROCESSED_TRAIN_PATH = train_path_out
        self.PREPROCESSED_TEST_PATH = test_path_out
        
        self.MODEL_PATH = "./unet/unet_3D/my_3d_model.h5"
        self.LOG_DIR = "unet\\unet_3d\\logs3d\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.VALIDATION_SPLIT = 0.1

        self.epochs_measured = 0

        self.ones = np.ones( ( self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE, 1 ), dtype=np.uint8 )

    def mask_to_classes( self, mask ):
        classes = np.zeros( ( self.NUM_CLASSES, self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE ), dtype=np.bool )
        for i, ( color, _ ) in enumerate( self.CLASSES ):
            curr_class_mask = np.all( np.equal( mask, color * self.ones ), axis = -1 )
            # plt.show()
            # imshow( curr_class_mask )
            classes[ i ] = curr_class_mask
        # Swap classes' axes from ( n, x, y, z ) to ( x, y, z, n )
        classes = np.swapaxes( classes, 0, 3 )
        classes = np.swapaxes( classes, 0, 2 )
        classes = np.swapaxes( classes, 0, 1 )
        return classes

    def masks_to_classes(self, masks):
        classes_masks = np.zeros( ( len( masks ), self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE, self.NUM_CLASSES ), dtype=np.bool)
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
        self.train_images = np.zeros( ( len( train_ids ), self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE, 1 ), dtype=np.uint8)
        self.train_masks  = np.zeros( ( len( train_ids ), self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE, self.NUM_CLASSES ), dtype=np.bool)

        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
            name = id_.split('.')[0]

            img = myUtils.load_and_prepare_nifti_image(self.PREPROCESSED_TRAIN_PATH + "images/" + id_)
            self.train_images[n] = img

            mask = myUtils.load_and_prepare_nifti_image(self.PREPROCESSED_TRAIN_PATH + 'masks/' + id_)
            self.train_masks[ n ] = self.mask_to_classes(mask)

    def load_testing_images( self ):
        print( "Read testing images from the disk" )

        # Get test IDs
        # Returns a list of file names in the testing path
        test_ids = next( os.walk( self.PREPROCESSED_TEST_PATH + "images/" ) )[2]

        # Get and resize train images and masks
        self.test_images = np.zeros( ( len( test_ids ), self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE, 1 ), dtype=np.uint8)
        self.test_masks  = np.zeros( ( len( test_ids ), self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE, self.NUM_CLASSES ), dtype=np.bool)

        for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
            name = id_.split('.')[0]

            img = myUtils.load_and_prepare_nifti_image(self.PREPROCESSED_TEST_PATH + "images/" + id_)
            self.test_images[n] = img

            if self.IS_TEST_DATA_LABELED:
                mask = myUtils.load_and_prepare_nifti_image(self.PREPROCESSED_TEST_PATH + 'masks/' + id_)
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
        inputs = Input( ( self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE, 1 ) )
        s = Lambda(lambda x: x / 255) (inputs)

        c1 = Conv3D(8, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
        c1 = Dropout(0.1) (c1)
        c1 = Conv3D(8, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
        p1 = MaxPooling3D((2, 2, 2)) (c1)

        c2 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
        c2 = Dropout(0.1) (c2)
        c2 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
        p2 = MaxPooling3D((2, 2, 2)) (c2)

        c3 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
        c3 = Dropout(0.2) (c3)
        c3 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
        p3 = MaxPooling3D((2, 2, 2)) (c3)

        c4 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
        c4 = Dropout(0.2) (c4)
        c4 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
        p4 = MaxPooling3D(pool_size=(2, 2, 2)) (c4)

        c5 = Conv3D(256, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
        c5 = Dropout(0.3) (c5)
        c5 = Conv3D(256, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

        u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
        c6 = Dropout(0.2) (c6)
        c6 = Conv3D(128, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

        u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
        c7 = Dropout(0.2) (c7)
        c7 = Conv3D(64, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

        u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
        c8 = Dropout(0.1) (c8)
        c8 = Conv3D(32, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

        u9 = Conv3DTranspose(8, (2, 2, 2), strides=(2, 2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=-1)
        c9 = Conv3D(8, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
        c9 = Dropout(0.1) (c9)
        c9 = Conv3D(8, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

        outputs = Conv3D(self.NUM_CLASSES, (1, 1, 1), activation='softmax') (c9)

        self.model = Model(inputs=[inputs], outputs=[outputs])
        #self.model.compile(optimizer='adam', loss=iou_coef_loss, metrics=[ iou_coef_loss, dice_coef_loss, "accuracy" ])
        self.model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[ iou_coef, dice_coef, iou_coef_loss, dice_coef_loss, "accuracy" ] )
        self.model.summary()
        
    def fit_model( self, epochs = 50 ):
        print( "Fit model" )
        
        # Fit model

        earlystopper = EarlyStopping(patience=5, verbose=1)
        checkpointer = ModelCheckpoint(self.MODEL_PATH, verbose=1, save_best_only=True, monitor="val_iou_coef_loss")
        # checkpointer = ModelCheckpoint( self.MODEL_PATH, verbose=1, save_best_only=True )
        tensorboard = TensorBoard(log_dir=self.LOG_DIR, histogram_freq=1)

        self.model.fit(self.train_images, self.train_masks, validation_split=self.VALIDATION_SPLIT,
                       batch_size=1, epochs=epochs,
                       callbacks=[earlystopper, checkpointer, tensorboard])

        self.epochs_measured += epochs

    # prediction has the shape x y z m
    def _prediction_to_mask( self, prediction ):
        dims = prediction.shape[:3] + (1,)
        mask = np.zeros( dims, dtype=np.uint8 )
        idxs = np.argmax( prediction, axis=-1 )
        for cn, ( k, _ ) in enumerate( self.CLASSES ):
            mask[ idxs == cn ] = k
        return mask

    # predictions has the shape n x y z m
    def _predictions_to_mask( self, predictions ):
        masks = np.zeros( predictions.shape[:-1] + (1,), dtype=np.uint8 )
        for i, img in enumerate( predictions ):
            masks[ i ] = self._prediction_to_mask( img )
        return masks

    def predict_volume( self, img ):
        original_size = img.shape[:]
        upscale_factor=(original_size[0]/self.IMG_SIZE,original_size[1]/self.IMG_SIZE,original_size[2]/self.IMG_SIZE)

        img = img * ( 255 / img.max() )

        resized_data = resize(img, (self.IMG_SIZE,self.IMG_SIZE,self.IMG_SIZE), mode='edge', preserve_range=True, order = 1, anti_aliasing = True)
        resized_data = resized_data.astype(np.uint8)

        resized_data = np.array([np.expand_dims( resized_data, axis=-1 )])

        preds = self.model.predict(resized_data, verbose=1)

        pred_resized = rescale(preds[0],upscale_factor,multichannel=True, mode='edge', preserve_range=True, order = 1, anti_aliasing = False)
        generated_mask = self._prediction_to_mask(pred_resized)
        generated_mask = np.squeeze(generated_mask,-1)
        #generated_mask_resized = resize(generated_mask, original_size, mode='edge', preserve_range=True, order = 0, anti_aliasing = False )

        return generated_mask

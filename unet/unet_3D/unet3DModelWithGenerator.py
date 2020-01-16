import os
import random
import warnings
import datetime

from skimage.transform import resize, rescale

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling3D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from unet.unet_3D.Nifti3DDataGenerator import *

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


class Unet3DModelWithGenerator:
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
        self.LOG_DIR = "unet\\unet_3D\\logs3d\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.VALIDATION_SPLIT = 0.125

        self.epochs_measured = 0

        self.predict_only = train_path_in is None           \
                            or test_path_in is None         \
                            or train_path_out is None       \
                            or test_path_out is None        \
                            or test_data_labeled is None

        if self.predict_only:
            return

        # Get train IDs
        # Returns a list of file names in the training path
        train_ids = next( os.walk( self.PREPROCESSED_TRAIN_PATH + "images/" ) )[2]
        np.random.shuffle(train_ids)
        validation_ids = train_ids[:int(len(train_ids)*self.VALIDATION_SPLIT)]
        learn_ids = train_ids[int(len(train_ids)*self.VALIDATION_SPLIT):]

        # Get test IDs
        # Returns a list of file names in the testing path
        test_ids = next( os.walk( self.PREPROCESSED_TEST_PATH + "images/" ) )[2]

        self.learning_generator = Nifti3DDataGenerator(learn_ids,
                                                       self.PREPROCESSED_TRAIN_PATH + "images/",
                                                       self.PREPROCESSED_TRAIN_PATH + "masks/",
                                                       self.CLASSES,
                                                       self.IMG_SIZE,
                                                       True)
        self.validation_generator = Nifti3DDataGenerator(validation_ids,
                                                         self.PREPROCESSED_TRAIN_PATH + "images/",
                                                         self.PREPROCESSED_TRAIN_PATH + "masks/",
                                                         self.CLASSES,
                                                         self.IMG_SIZE,
                                                         True)
        self.test_generator = Nifti3DDataGenerator(test_ids,
                                                   self.PREPROCESSED_TEST_PATH + "images/",
                                                   self.PREPROCESSED_TEST_PATH + "masks/",
                                                   self.CLASSES,
                                                   self.IMG_SIZE,
                                                   self.IS_TEST_DATA_LABELED)

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

        c1 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
        c1 = Dropout(0.1) (c1)
        c1 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
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
        c9 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
        c9 = Dropout(0.1) (c9)
        c9 = Conv3D(16, (3, 3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

        outputs = Conv3D(self.NUM_CLASSES, (1, 1, 1), activation='softmax') (c9)

        self.model = Model(inputs=[inputs], outputs=[outputs])
        #self.model.compile(optimizer='adam', loss=iou_coef_loss, metrics=[ iou_coef_loss, dice_coef_loss, "accuracy" ])
        self.model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[ iou_coef, dice_coef, iou_coef_loss, dice_coef_loss, "accuracy" ] )
        self.model.summary()
        
    def fit_model( self, epochs = 50 ):
        print( "Fit model" )

        if self.predict_only:
            print("Not possible in predict only mode")
            return

        # Fit model
        earlystopper = EarlyStopping(patience=5, verbose=1, monitor="iou_coef_loss")
        checkpointer = ModelCheckpoint(self.MODEL_PATH, verbose=1, save_best_only=True, monitor="iou_coef_loss")
        # checkpointer = ModelCheckpoint( self.MODEL_PATH, verbose=1, save_best_only=True )

        self.model.fit_generator(generator=self.learning_generator,epochs=epochs,
                                 callbacks=[earlystopper, checkpointer],
                                 validation_data=self.validation_generator)

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

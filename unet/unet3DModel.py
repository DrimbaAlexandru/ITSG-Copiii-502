import os
import random
import warnings
import datetime

import numpy as np
import nibabel as nib

from tqdm import tqdm
from skimage.io import imread, imsave, imshow
from skimage.transform import resize
from skimage.color import gray2rgb, rgb2gray

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling3D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K

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

class Unet_3d_model:
    def __init__( self,
                  size,
                  train_path_in,
                  test_path_in,
                  train_path_out,
                  test_path_out,
                  test_data_labeled,
                  classes = [ ( (0), "Background" ), ( (255), "Class 1" ) ] ):
                 
        self.IMG_SIZE = size
        self.TRAIN_PATH = train_path_in
        self.TEST_PATH = test_path_in
        self.CLASSES = classes
        self.NUM_CLASSES = len( classes )
        self.IS_TEST_DATA_LABELED = test_data_labeled

        self.PREPROCESSED_TRAIN_PATH = train_path_out
        self.PREPROCESSED_TEST_PATH = test_path_out
        
        self.MODEL_PATH = "./unet/my_3d_model.h5"
        self.LOG_DIR = "logs3d\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(self.LOG_DIR)

        self.VALIDATION_SPLIT = 0.1

        self.metrics = {}
        self.epochs_measured = 0
        self.tensorboard = TensorBoard(log_dir=self.LOG_DIR, histogram_freq=1)

        self.ones = np.ones( ( self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE, 1 ), dtype=np.uint8 )

    def _load_image( self, path ):
        proxy_img = nib.load( path )
        canonical_img = nib.as_closest_canonical(proxy_img)

        image_data = canonical_img.get_fdata()
        min_max = ( image_data.min(), image_data.max() )

        image_data = image_data * ( 255 / min_max[1] )
        image_data = image_data.astype(np.uint8)
        image_data = np.expand_dims( image_data, axis=-1 )

        return image_data

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

            img = self._load_image( self.PREPROCESSED_TRAIN_PATH + "images/" + id_ )
            self.train_images[n] = img

            mask = self._load_image( self.PREPROCESSED_TRAIN_PATH + 'masks/' + id_ )
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

            img = self._load_image( self.PREPROCESSED_TEST_PATH + "images/" + id_ )
            self.test_images[n] = img

            if self.IS_TEST_DATA_LABELED:
                mask = self._load_image( self.PREPROCESSED_TEST_PATH + 'masks/' + id_ )
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

        self.model.fit(self.train_images, self.train_masks, validation_split=self.VALIDATION_SPLIT,
                       batch_size=1, epochs=epochs,
                       callbacks=[earlystopper, checkpointer, self.tensorboard])

        self.epochs_measured += epochs

    # prediction has the shape x y z
    def _prediction_to_mask( self, prediction ):
        mask = np.zeros( ( self.IMG_SIZE, self.IMG_SIZE,  self.IMG_SIZE, 1 ), dtype=np.uint8 )
        idxs = np.argmax( prediction, axis=-1 )
        for cn, ( k, _ ) in enumerate( self.CLASSES ):
            mask[ idxs == cn ] = k
        return mask

    # predictions has the shape n x y z
    def _predictions_to_mask( self, predictions ):
        masks = np.zeros( predictions.shape[:], dtype=np.uint8 )
        for i, img in enumerate( predictions ):
            masks[ i ] = self._prediction_to_mask( img )
        return masks

    def predict_volume( self, img ):
        original_size = img.shape[:3]
        print(original_size)

        img = img * ( 255 / img.max() )
        imgs = np.zeros( ( 1 , self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE, 1 ), dtype=np.uint8)

        resized_data = resize(img, (self.IMG_SIZE,self.IMG_SIZE,self.IMG_SIZE), mode='edge', preserve_range=True, order = 1, anti_aliasing = False)
        resized_data = resized_data.astype(np.uint8)
        imgs[0] = resized_data
        preds = self.model.predict(imgs,  verbose = 1)

        generated_masks = self._predictions_to_mask(preds)
        print(generated_masks.shape)

        generated_mask = generated_masks[0]
        print(generated_mask.shape)

        list = []
        for i in range(self.IMG_SIZE):
            for j in range(self.IMG_SIZE):
                for k in range(self.IMG_SIZE):
                    list.append(generated_mask[i,j,k].tolist())

        palette = sort_and_deduplicate(list)
        generated_mask_resized = resize(gm, original_size, mode='edge', preserve_range=True, order = 0, anti_aliasing = False )

        return generated_mask_resized

    def uniq(lst):
        last = object()
        for item in lst:
            if item == last:
                continue
            yield item
            last = item

    def sort_and_deduplicate(l):
        return list(uniq(sorted(l, reverse=True)))

    def evaluate_model(self):

        input_learn = self.train_images[int(self.train_images.shape[0] * self.VALIDATION_SPLIT ):]
        input_val   = self.train_images[:int(self.train_images.shape[0] * self.VALIDATION_SPLIT )]

        masks_learn = self.train_masks[int(self.train_images.shape[0] * self.VALIDATION_SPLIT ):]
        masks_val  = self.train_masks[:int(self.train_images.shape[0] * self.VALIDATION_SPLIT )]

        metrics_learn = self.model.evaluate(input_learn,masks_learn,batch_size=1)
        metrics_val = self.model.evaluate(input_val,masks_val,batch_size=1)

        self.metrics[self.epochs_measured]=[]
        self.metrics[self.epochs_measured].append(metrics_learn[1:3])
        self.metrics[self.epochs_measured].append(metrics_val[1:3])

        if( self.IS_TEST_DATA_LABELED ):
            metrics_test = self.model.evaluate(self.test_images,self.test_masks,batch_size=1)
            self.metrics[self.epochs_measured].append(metrics_test[1:3])


    def write_model_metrics(self):
        resultsFile = open(self.LOG_DIR + "\\results.txt", "w")

        resultsFile.write("Number of learning samples: " + str( self.train_images.shape[0] * ( 1 - self.VALIDATION_SPLIT ) ) )
        resultsFile.write("\nNumber of validation samples: " + str( self.train_images.shape[0] * self.VALIDATION_SPLIT ) )
        if( self.IS_TEST_DATA_LABELED ):
            resultsFile.write("\nNumber of testing samples: " + str( len(self.test_images) ) )

        resultsFile.write("\nLearning results:" )
        resultsFile.write("\n  IoU,   Dice\n")
        for epoch in range(0,self.epochs_measured+1):
            if epoch in self.metrics:
                for metric in self.metrics[epoch][0]:
                    resultsFile.write( str( metric )+ ", " )
                resultsFile.write("\n")
        resultsFile.write("\n")

        resultsFile.write("\nValidation results:" )
        resultsFile.write("\n  IoU,   Dice\n")
        for epoch in range(0,self.epochs_measured+1):
            if epoch in self.metrics:
                for metric in self.metrics[epoch][1]:
                    resultsFile.write( str( metric )+ ", " )
                resultsFile.write("\n")
        resultsFile.write("\n")

        if( self.IS_TEST_DATA_LABELED ):
            resultsFile.write("\nTesting results:" )
            resultsFile.write("\n  IoU,   Dice\n")
            for epoch in range(0,self.epochs_measured+1):
                if epoch in self.metrics:
                    for metric in self.metrics[epoch][2]:
                        resultsFile.write( str( metric )+ ", " )
                    resultsFile.write("\n")
            resultsFile.write("\n")

            resultsFile.close()

from unet.preprocessOneClass import OneClassPreprocessor
from unet.preprocessNifti import NIfTIPreprocessor
from unet.unetModel import Unet_model
import numpy as np
from skimage.io import imread, imsave, imshow

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = './input/NIfTI/NIfTIs/training/'
TEST_PATH = './input/NIfTI/NIfTIs/testing/'
PREPROCESSED_TRAIN_PATH = "./input/NIfTI/training/"
PREPROCESSED_TEST_PATH = "./input/NIfTI/testing/"
RESULTS_PATH = "./output/"
TEST_DATA_LABELED = True

needs_preprocess = False;


def load_model():
    if needs_preprocess:
        preprocessor = NIfTIPreprocessor( IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, TRAIN_PATH, TEST_PATH, PREPROCESSED_TRAIN_PATH, PREPROCESSED_TEST_PATH, TEST_DATA_LABELED )
        preprocessor.preprocess()
    
    model = Unet_model( IMG_WIDTH,
                    IMG_HEIGHT,
                    IMG_CHANNELS,
                    TRAIN_PATH,
                    TEST_PATH,
                    PREPROCESSED_TRAIN_PATH,
                    PREPROCESSED_TEST_PATH,
                    TEST_DATA_LABELED,
                    [ ( ( 0, 0, 0 ), "Background" ), ( ( 127, 127, 127 ), "Ventricular Myocardum" ), ( ( 255, 255, 255 ), "Blood Pool" ) ] )
    #model.load_training_images()
    #model.load_testing_images()    
    #model.create_model()
    model.load_model()
    #model.fit_model( 10 )
    #model.save_model()
    #model.predict_from_model()
    #model.evaluate_model()

    # Adaugare metrici
    # Conferinta imogen in 8.11, de la 9 la 11
    return model
from preprocessOneClass import OneClassPreprocessor
from unetModel import Unet_model


# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = './input/stage1_train/'
TEST_PATH = './input/stage1_test/'
PREPROCESSED_TRAIN_PATH = "./input/training/"
PREPROCESSED_TEST_PATH = "./input/testing/"

needs_preprocess = False

if needs_preprocess:
    preprocessor = OneClassPreprocessor( IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, TRAIN_PATH, TEST_PATH, PREPROCESSED_TRAIN_PATH, PREPROCESSED_TEST_PATH )
    preprocessor.preprocess()
    
model = Unet_model( IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, TRAIN_PATH, TEST_PATH, PREPROCESSED_TRAIN_PATH, PREPROCESSED_TEST_PATH )
model.load_images()
#model.create_model()
model.load_model()
#model.fit_model( 10 )
model.predict_from_model()
model.save_model()
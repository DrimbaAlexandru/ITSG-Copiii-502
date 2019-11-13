from preprocessOneClass import OneClassPreprocessor
from preprocessNifti import NIfTIPreprocessor
from unetModel import Unet_model


# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = './input/NIfTI/NIfTIs/training/'
TEST_PATH = './input/NIfTI/NIfTIs/testing/'
PREPROCESSED_TRAIN_PATH = "./input/NIfTI/training/"
PREPROCESSED_TEST_PATH = "./input/NIfTI/testing/"

needs_preprocess = True

if needs_preprocess:
    preprocessor = NIfTIPreprocessor( IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, TRAIN_PATH, TEST_PATH, PREPROCESSED_TRAIN_PATH, PREPROCESSED_TEST_PATH )
    preprocessor.preprocess()
    
model = Unet_model( IMG_WIDTH,
                    IMG_HEIGHT,
                    IMG_CHANNELS,
                    TRAIN_PATH,
                    TEST_PATH,
                    PREPROCESSED_TRAIN_PATH,
                    PREPROCESSED_TEST_PATH,
                    [ ( ( 0, 0, 0 ), "Bg" ), ( ( 127, 127, 127 ), "Ventricular Myocardum" ), ( ( 255, 255, 255 ), "Blood Pool" ) ] )
model.load_images()
model.create_model()
#model.load_model()
model.fit_model( 5 )
model.predict_from_model()
model.save_model()

# Adaugare metrici
# Conferinta imogen in 8.11, de la 9 la 11
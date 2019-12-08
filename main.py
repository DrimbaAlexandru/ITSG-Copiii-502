from unet.preprocessNifti import NIfTIPreprocessor
from unet.unetModel import Unet_model
import nibabel as nib

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = './unet/input/NIfTI/NIfTIs/training/'
TEST_PATH = './unet/input/NIfTI/NIfTIs/testing/'
PREPROCESSED_TRAIN_PATH = "./unet/input/NIfTI/training/"
PREPROCESSED_TEST_PATH = "./unet/input/NIfTI/testing/"
TEST_DATA_LABELED = True

needs_preprocess = False

if needs_preprocess:
    preprocessor = NIfTIPreprocessor( IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, TRAIN_PATH, TEST_PATH, PREPROCESSED_TRAIN_PATH, PREPROCESSED_TEST_PATH, TEST_DATA_LABELED )
    preprocessor.preprocess()

model = Unet_model( IMG_CHANNELS,
                    TRAIN_PATH,
                    TEST_PATH,
                    PREPROCESSED_TRAIN_PATH,
                    PREPROCESSED_TEST_PATH,
                    TEST_DATA_LABELED,
                    [ ( ( 0, 0, 0 ), "Background" ), ( ( 127, 127, 127 ), "Ventricular Myocardum" ), ( ( 255, 255, 255 ), "Blood Pool" ) ] )
model.load_training_images()
model.load_testing_images()
model.create_model()
#model.load_model()
for i in range( 0, 10 ):
    model.fit_model( 1 )
    model.save_model()
    #model.predict_from_model()
    model.evaluate_model()
model.write_model_metrics()
#result = model.predict_volume( image_data )

# Adaugare metrici
# Conferinta imogen in 8.11, de la 9 la 11
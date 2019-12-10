from unet.preprocessNiftis import NIfTIsPreprocessor
from unet.unetModel import Unet_model
from unet.unet3DModel import Unet_3d_model

import random

# Set some parameters
IMG_SIZE = 96
IMG_CHANNELS = 3
TRAIN_PATH = './unet/input/NIfTI/NIfTIs/training/'
TEST_PATH = './unet/input/NIfTI/NIfTIs/testing/'
PREPROCESSED_TRAIN_PATH = "./unet/input/NIfTI/training/"
PREPROCESSED_TEST_PATH = "./unet/input/NIfTI/testing/"
TEST_DATA_LABELED = True

needs_preprocess = True

if needs_preprocess:
    preprocessor = NIfTIsPreprocessor( IMG_SIZE, TRAIN_PATH, TEST_PATH, PREPROCESSED_TRAIN_PATH, PREPROCESSED_TEST_PATH, TEST_DATA_LABELED )

    # for i in range( 0, 1 ):
    #     preprocessor.handicapeaza("D:/git/ITSG-Copiii-502/unet/input/NIfTI/NIfTIs/Training dataset/training_axial_full_pat"+str(i)+".nii.gz",
    #                               "D:/git/ITSG-Copiii-502/unet/input/NIfTI/NIfTIs/training/training_axial_full_pat"+str(i)+"_0.nii.gz",
    #                               coords1[i],
    #                               coords2[i],
    #                               (0,random.random()*40-20,random.random()*40-20))

    preprocessor.preprocess()

model = Unet_3d_model( IMG_SIZE,
                       TRAIN_PATH,
                       TEST_PATH,
                       PREPROCESSED_TRAIN_PATH,
                       PREPROCESSED_TEST_PATH,
                       TEST_DATA_LABELED,
                       [ ( (0), "Background" ), ( (127), "Ventricular Myocardum" ), ( (255), "Blood Pool" ) ] )

model.load_training_images()
model.load_testing_images()
model.create_model()
#model.load_model()
for i in range( 0, 1 ):
    model.fit_model( 100 )
    model.save_model()
    #model.predict_from_model()
    model.evaluate_model()
model.write_model_metrics()
#result = model.predict_volume( image_data )

# Adaugare metrici
# Conferinta imogen in 8.11, de la 9 la 11
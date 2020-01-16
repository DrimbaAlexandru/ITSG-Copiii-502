from utils.NIfTI3DPreprocessor import NIfTI3DPreprocessor
from unet.unet_3D.unet3DModelWithGenerator import Unet3DModelWithGenerator

import numpy as np
import random

import utils.utils as myUtils

# Set some parameters
IMG_SIZE = 64
IMG_CHANNELS = 3
TRAIN_PATH = './input/NIfTI/NIfTIs/training/'
TEST_PATH = './input/NIfTI/NIfTIs/testing/'
PREPROCESSED_TRAIN_PATH = "./input/NIfTI/training/"
PREPROCESSED_TEST_PATH = "./input/NIfTI/testing/"
TEST_DATA_LABELED = True

needs_preprocess = False
needs_augmentation = False

images_coords = {
    "training_axial_full_pat0" : [((0, 130, 135), (150, 130, 205), (0, 0, 0))],

    "training_axial_full_pat1" : [((0, 140, 110), (145, 120, 210), (0, 0, 0))],

    "training_axial_full_pat2" : [((0, 120, 85), (180, 120, 210), (0, 0, 0))],

    "training_axial_full_pat3" : [((0, 120, 110), (150, 135, 170), (0, 0, 0))],

    "training_axial_full_pat4" : [((0, 95, 105), (125, 80, 105), (0, 0, 0))],

    "training_axial_full_pat5" : [((0, 95, 75), (165, 85, 110), (0, 0, 0))],

    "training_axial_full_pat6" : [((0, 215, 185), (140, 120, 180), (0, 0, 0))],

    "training_axial_full_pat7" : [((0, 200, 180), (185, 155, 240), (0, 0, 0))],

    "training_axial_full_pat8" : [((0, 140, 125), (140, 125, 190), (0, 0, 0))],

    "training_axial_full_pat9" : [((0, 135, 100), (175, 150, 215), (0, 0, 0))]
}

preprocessor = NIfTI3DPreprocessor(IMG_SIZE, TRAIN_PATH, TEST_PATH, PREPROCESSED_TRAIN_PATH, PREPROCESSED_TEST_PATH, TEST_DATA_LABELED)

if needs_augmentation:
    for img_name in images_coords:
        for aug in range(3):
            new_corner_offset = np.array(( 0,
                                           int(random.random()*10 - 3),
                                           int(random.random()*10 - 3)))
            new_lengths_offset = np.array(( int(random.random()*10 - 3),
                                            int(random.random()*10 - 3),
                                            int(random.random()*10 - 3)))
            el = ( np.array(images_coords[img_name][0][0])-new_corner_offset,
                   np.array(images_coords[img_name][0][1]) + new_corner_offset + new_lengths_offset,
                   None )
            images_coords[img_name].append(el)

        for i, (top_left, lengths, skew) in enumerate(images_coords[img_name]):
            print("Processing variation "+str(i+1)+" on "+img_name)
            dither = skew is None
            if skew is None:
                skew = ( random.random()*0.2-0.1, random.random()*0.2-0.1, random.random()*0.2-0.1 )

            # Distort the training image
            preprocessor.distort("D:/git/ITSG-Copiii-502/unet/dataset/Training dataset/"+img_name+".nii.gz",
                                 TRAIN_PATH+"images/"+img_name+"_"+str(i)+".nii.gz",
                                 top_left,
                                 lengths,
                                 skew,
                                 dither)

            # Distort its maks using the same parameters
            preprocessor.distort("D:/git/ITSG-Copiii-502/unet/dataset/Ground truth/"+img_name+"-label.nii.gz",
                                 TRAIN_PATH+"masks/"+img_name+"_"+str(i)+"-label.nii.gz",
                                 top_left,
                                 lengths,
                                 skew,
                                 False)

if needs_preprocess:
    preprocessor.preprocess()

model = Unet3DModelWithGenerator([ ( (0), "Background" ), ( (127), "Ventricular Myocardum" ), ( (255), "Blood Pool" ) ],
                                 IMG_SIZE,
                                 TRAIN_PATH,
                                 TEST_PATH,
                                 PREPROCESSED_TRAIN_PATH,
                                 PREPROCESSED_TEST_PATH,
                                 TEST_DATA_LABELED)
metrics = {}

model.create_model()
#model.load_model()
for i in range( 0, 10 ):
    model.fit_model( 5 )
    model.save_model()
    #model.predict_from_model()
    myUtils.evaluate_model_generator(model,metrics)
myUtils.write_model_metrics(model,metrics)
#result = model.predict_volume( image_data )

# Adaugare metrici
# Conferinta imogen in 8.11, de la 9 la 11
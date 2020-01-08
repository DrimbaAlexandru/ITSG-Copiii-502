import os
import warnings

import numpy as np
import utils.utils as myUtils

from skimage.transform import resize
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


class NIfTI3DPreprocessor:
    def __init__(self, size, train_path_in, test_path_in, train_path_out, test_path_out, test_labeled=False ):
        self.IMG_SIZE = size
        self.TRAIN_PATH = train_path_in
        self.TEST_PATH = test_path_in
        self.TEST_LABELED = test_labeled

        self.PREPROCESSED_TRAIN_PATH = train_path_out
        self.PREPROCESSED_TEST_PATH = test_path_out

        os.makedirs(self.TRAIN_PATH + 'images/', exist_ok=True)
        os.makedirs(self.TRAIN_PATH + 'masks/', exist_ok=True)
        os.makedirs(self.TEST_PATH + 'images/', exist_ok=True)
        os.makedirs(self.TEST_PATH + 'masks/', exist_ok=True)

        os.makedirs(self.PREPROCESSED_TRAIN_PATH + 'images/', exist_ok=True)
        os.makedirs(self.PREPROCESSED_TRAIN_PATH + 'masks/', exist_ok=True)
        os.makedirs(self.PREPROCESSED_TEST_PATH + 'images/', exist_ok=True)
        os.makedirs(self.PREPROCESSED_TEST_PATH + 'masks/', exist_ok=True)

    def distort(self, img_path_in, img_path_out, top_left, lengths, skew, dither = False):
        image_data, affine = myUtils.load_nifti_image(img_path_in)
        shape = image_data.shape[:]
        src_coords = np.zeros((3))

        if top_left is None:
            top_left = ( 0, 0, 0 )
        if lengths is None:
            lengths = np.array(shape) - np.array(top_left)

        img_data_out = np.zeros(lengths)

        for x in tqdm(range(0, lengths[0]), total=lengths[0]):
            for y in range(0, lengths[1]):
                for z in range(0, lengths[2]):
                    src_coords[0] = x + top_left[0] + skew[0] * (y + z)
                    src_coords[1] = y + top_left[1] + skew[1] * (x + z)
                    src_coords[2] = z + top_left[2] + skew[2] * (x + y)

                    if(0 <= src_coords[0] < shape[0] - 1
                   and 0 <= src_coords[1] < shape[1] - 1
                   and 0 <= src_coords[2] < shape[2] - 1):
                        src_coords_ints = np.floor(src_coords).astype(np.int16)
                        img_data_out[x][y][z] = image_data[src_coords_ints[0]][src_coords_ints[1]][src_coords_ints[2]]
                        src_coords_dec = src_coords - src_coords_ints
                        if dither and ( src_coords_dec[0]>=0.7 or src_coords_dec[1]>=0.7 or src_coords_dec[2]>=0.7 ):
                            img_data_out[x][y][z] *= 1 - np.sum(src_coords_dec)/3
                            img_data_out[x][y][z] += src_coords_dec[0]/3 * image_data[src_coords_ints[0] + 1][ src_coords_ints[1], src_coords_ints[2]] + \
                                                     src_coords_dec[1]/3 * image_data[src_coords_ints[0]][src_coords_ints[1] + 1][src_coords_ints[2]] + \
                                                     src_coords_dec[2]/3 * image_data[src_coords_ints[0]][src_coords_ints[1]][src_coords_ints[2] + 1]

        myUtils.save_nifti_image(img_data_out, affine, img_path_out)

    def preprocess( self ):
        # Get train and test IDs
        # Returns a list of image file names in the training and testing paths
        # Label files are expected to be image_name -label
        train_image_ids = next(os.walk(self.TRAIN_PATH + 'images/'))[2]
        test_image_ids = next(os.walk(self.TEST_PATH + 'images/'))[2]

        print('Getting and resizing training images and masks... ', flush=True)
        for n, id_ in enumerate(train_image_ids):
            # Process images
            print( "Exporting image %d/%d" % ( n+1, len(train_image_ids) ), flush=True )

            name = id_.split('.')[0]
            image_data, affine = myUtils.load_nifti_image(self.TRAIN_PATH + 'images/' + id_)
            image_data = resize(image_data, (self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE), mode='edge', preserve_range=True )

            myUtils.save_nifti_image(image_data, affine, self.PREPROCESSED_TRAIN_PATH + "images/" + id_)

            # Process masks
            image_data, affine = myUtils.load_nifti_image(self.TRAIN_PATH + 'masks/' + name + "-label.nii.gz")
            image_data = resize(image_data, (self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE), mode='edge', preserve_range=True, order = 0, anti_aliasing = False )

            myUtils.save_nifti_image(image_data, affine, self.PREPROCESSED_TRAIN_PATH + "masks/" + id_)

        print('Getting and resizing testing images and masks... ', flush=True)
        for n, id_ in enumerate(test_image_ids):
            # Process images
            print( "Exporting image %d/%d" % ( n+1, len(test_image_ids) ), flush=True )

            name = id_.split('.')[0]
            image_data, affine = myUtils.load_nifti_image(self.TEST_PATH + 'images/' + id_)
            image_data = resize(image_data, (self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE), mode='edge', preserve_range=True )

            myUtils.save_nifti_image(image_data, affine, self.PREPROCESSED_TEST_PATH + "images/" + id_)

            if not self.TEST_LABELED:
                continue

            # Process masks
            image_data, affine = myUtils.load_nifti_image(self.TEST_PATH + 'masks/' + name + "-label.nii.gz")
            image_data = resize(image_data, (self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE), mode='edge', preserve_range=True, order = 0, anti_aliasing = False )

            myUtils.save_nifti_image(image_data, affine, self.PREPROCESSED_TEST_PATH + "masks/" + id_)

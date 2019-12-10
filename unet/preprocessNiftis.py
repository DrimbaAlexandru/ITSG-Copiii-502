import os
import warnings

import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage.io import  imsave
from skimage.transform import resize
from skimage.color import gray2rgb

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


class NIfTIsPreprocessor:
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

    def handicapeaza(self, img_path_in, img_path_out, coord1, coord2, skew ):

        proxy_img = nib.load( img_path_in )
        canonical_img = nib.as_closest_canonical(proxy_img)
        lengths = ( coord2[0] - coord1[0], coord2[1] - coord1[1], coord2[2] - coord1[2] )
        image_data = canonical_img.get_fdata()
        img_data_out = np.zeros(lengths)

        for i in range( 0, lengths[0] ):
            for j in range( 0, lengths[1] ):
                for k in range( 0, lengths[2] ):
                    img_data_out[i][j][k] = image_data [ i + coord1[0] + int(skew[0])//lengths[0]]   \
                                                       [ j + coord1[1] + int(skew[1])//lengths[1]]   \
                                                       [ k + coord1[2] + int(skew[2])//lengths[2]]

        img = nib.Nifti1Image(img_data_out, canonical_img.affine)
        img.to_filename( img_path_out )
        nib.save(img, img_path_out )


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
            proxy_img = nib.load( self.TRAIN_PATH + 'images/' + id_)
            canonical_img = nib.as_closest_canonical(proxy_img)
            
            image_data = canonical_img.get_fdata()
            image_data = resize(image_data, (self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE), mode='edge', preserve_range=True )

            img = nib.Nifti1Image(image_data, canonical_img.affine)
            img.to_filename(self.PREPROCESSED_TRAIN_PATH + "images/" + id_ )
            nib.save(img, self.PREPROCESSED_TRAIN_PATH + "images/" + id_ )

            # Process masks
            proxy_img = nib.load( self.TRAIN_PATH + 'masks/' + name + "-label.nii.gz")
            canonical_img = nib.as_closest_canonical(proxy_img)
            
            image_data = canonical_img.get_fdata()
            image_data = resize(image_data, (self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE), mode='edge', preserve_range=True, order = 0, anti_aliasing = False )

            img = nib.Nifti1Image(image_data, canonical_img.affine)
            img.to_filename(self.PREPROCESSED_TRAIN_PATH + "masks/" + id_ )
            nib.save(img, self.PREPROCESSED_TRAIN_PATH + "masks/" + id_ )

        print('Getting and resizing testing images and masks... ', flush=True)
        for n, id_ in enumerate(test_image_ids):
            # Process images
            print( "Exporting image %d/%d" % ( n+1, len(test_image_ids) ), flush=True )

            name = id_.split('.')[0]
            proxy_img = nib.load( self.TEST_PATH + 'images/' + id_)
            canonical_img = nib.as_closest_canonical(proxy_img)

            image_data = canonical_img.get_fdata()
            image_data = resize(image_data, (self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE), mode='edge', preserve_range=True )

            img = nib.Nifti1Image(image_data, canonical_img.affine)
            img.to_filename(self.PREPROCESSED_TEST_PATH + "images/" + id_ )
            nib.save(img, self.PREPROCESSED_TEST_PATH + "images/" + id_ )

            if not self.TEST_LABELED:
                continue

            # Process masks
            proxy_img = nib.load( self.TEST_PATH + 'masks/' + name + "-label.nii.gz")
            canonical_img = nib.as_closest_canonical(proxy_img)

            image_data = canonical_img.get_fdata()
            image_data = resize(image_data, (self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE), mode='edge', preserve_range=True, order = 0, anti_aliasing = False )

            img = nib.Nifti1Image(image_data, canonical_img.affine)
            img.to_filename(self.PREPROCESSED_TEST_PATH + "masks/" + id_ )
            nib.save(img, self.PREPROCESSED_TEST_PATH + "masks/" + id_ )




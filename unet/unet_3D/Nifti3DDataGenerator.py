import numpy as np
from keras.utils import Sequence

import utils.utils as myUtils


class Nifti3DDataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, image_path, mask_path,
                 classes, img_size,
                 to_fit=True, batch_size=2, shuffle=True):
        """Initialization
        :param list_IDs: list of all image filenames to use in the generator
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param classes: list of (color, name) tuples for each class
        :param img_size: image dimensions
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.image_path = image_path
        self.mask_path = mask_path
        self.no_classes = len(classes)
        self.classes = classes
        self.img_size = img_size
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

        self.ones = np.ones( ( self.img_size, self.img_size, self.img_size, 1 ), dtype=np.uint8 )

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_list_IDs = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = np.array([myUtils.load_and_prepare_nifti_image(self.image_path+id) for id in batch_list_IDs])

        if self.to_fit:
            y = np.array([self._mask_to_classes(myUtils.load_and_prepare_nifti_image(self.mask_path+id)) for id in batch_list_IDs])
            return X, y
        else:
            return X

    def get_item_count(self):
        return len(self.list_IDs)

    def _mask_to_classes( self, mask ):
        classes = np.zeros( ( self.no_classes, self.img_size, self.img_size, self.img_size ), dtype=np.bool )
        for i, ( color, _ ) in enumerate( self.classes ):
            curr_class_mask = np.all( np.equal( mask, color * self.ones ), axis = -1 )
            # plt.show()
            # imshow( curr_class_mask )
            classes[ i ] = curr_class_mask
        # Swap classes' axes from ( n, x, y, z ) to ( x, y, z, n )
        classes = np.swapaxes( classes, 0, 3 )
        classes = np.swapaxes( classes, 0, 2 )
        classes = np.swapaxes( classes, 0, 1 )
        return classes

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

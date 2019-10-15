import nibabel as nib
import matplotlib.pyplot as plt

class MRI_plot:
    _base_image_path = None
    _mask_image_path = None
    
    _base_image_data = None
    _mask_image_data = None
    
    _mask_transparency = 0.5
    
    _axial_pos = 0
    _saggital_pos = 0
    _coronal_pos = 0
    
    _plot_canvas = None
    _plot_axes = None
    
    def __init__( self, image_path, mask_path, transparency = 0.5 ):
        self.set_image_paths( image_path, mask_path, transparency )
        
        
    def set_image_paths( self, image_path, mask_path, transparency = 0.5 ):
        self._base_image_path = image_path
        self._mask_image_path = mask_path
        self._mask_transparency = max( min( transparency, 1 ), 0 )
        
        self.redraw()
        
        
    def redraw( self ):
        if( self._base_image_path != None ):
            proxy_img = nib.load( self._base_image_path )
            self._base_image_data = proxy_img.get_fdata()
            print self._base_image_data.shape
            
            self._axial_pos = self._base_image_data.shape[ 0 ] // 2
            self._saggital_pos = self._base_image_data.shape[ 1 ] // 2
            self._coronal_pos = self._base_image_data.shape[ 2 ] // 2
        else:
            self._base_image_data = None
        
        if( self._mask_image_path != None ):
            proxy_img = nib.load( self._mask_image_path )
            self._mask_image_data = proxy_img.get_fdata()
            print self._mask_image_data.shape
        else:
            self._mask_image_data = None
        
        self._remove_keymap_conflicts({'j', 'k'})
        self._display_current_frame()
        self._plot_canvas.canvas.mpl_connect('key_press_event', self._process_key)
        plt.show()
        
        
    def _display_current_frame( self ):
        if( self._plot_canvas == None ):
            self._plot_canvas, self._plot_axes = plt.subplots( nrows = 1, ncols = 3 )
        
        print self._axial_pos
        
        slices = [ self._base_image_data[ self._axial_pos, : , :     ],
                   self._base_image_data[ : , self._saggital_pos , : ],
                   self._base_image_data[ : , : , self._coronal_pos  ] ] 
                   
        for i, slice in enumerate(slices):
            self._plot_axes[ i ].imshow( slice.T, cmap="gray", origin="lower" )       
        self._plot_canvas.canvas.draw()
        
    def _move_slice( self, axial_delta, saggital_delta, coronal_delta ):
        self._axial_pos = self._axial_pos + axial_delta
        self._axial_pos = self._axial_pos % self._base_image_data.shape[ 0 ]
        
    def _remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def multi_slice_viewer(volume):
        _remove_keymap_conflicts({'j', 'k'})
        fig, ax = plt.subplots()
        ax.volume = volume
        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index])
        fig.canvas.mpl_connect('key_press_event', _process_key)

    def _process_key(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            self._move_slice( 1, 0, 0 )
        elif event.key == 'k':
            self._move_slice( -1, 0, 0 )
        
        self._display_current_frame()
        #fig.canvas.draw()

    def previous_slice(ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])

    def next_slice(ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])

# struct = nib.load('D:/git/ITSG/ITSG/Training dataset/training_axial_full_pat9.nii.gz')
# struct_arr = struct.get_data()
# struct_arr2 = struct_arr.T
# multi_slice_viewer(struct_arr2)
# plt.show()
import matplotlib

matplotlib.use('TkAgg')

import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.colors import Normalize


class MRI_plot:
    _base_image_path = ""
    _mask_image_path = ""
    
    _base_image_data = None
    _mask_image_data = None
    _image_min_max = ( 0, 1 )
    
    _mask_transparency = 0.333
    
    _axial_pos = 0
    _saggital_pos = 0
    _coronal_pos = 0
    
    _plot_canvas = None
    _plot_artists = []
    
    is_window_showing = False
    _is_mask_displayed = True
    
    def __init__( self, image_path, mask_path, transparency = None ):
        self.set_image_paths( image_path, mask_path, transparency )
        
        
    def set_image_paths( self, image_path, mask_path, transparency = None ):
        self._base_image_path = image_path
        self._mask_image_path = mask_path
        if( transparency is not None ):
            self._mask_transparency = max( min( transparency, 1 ), 0 )
        
        self.reload()
        
        
    def reload( self ):
        if( self._base_image_path != "" ):
            proxy_img = nib.load( self._base_image_path )
            canonical_img = nib.as_closest_canonical(proxy_img)
            
            self._base_image_data = canonical_img.get_fdata()
            self._image_min_max = ( self._base_image_data.min(), self._base_image_data.max() )
            
            print( self._base_image_data.shape)
            
            self._axial_pos = self._base_image_data.shape[ 0 ] // 2
            self._saggital_pos = self._base_image_data.shape[ 1 ] // 2
            self._coronal_pos = self._base_image_data.shape[ 2 ] // 2
        else:
            self._base_image_data = None
        
        if( self._mask_image_path != "" ):
            proxy_img = nib.load( self._mask_image_path )
            canonical_img = nib.as_closest_canonical(proxy_img)
            self._mask_image_data = canonical_img.get_fdata()
            
            if( self._mask_image_data.shape != self._base_image_data.shape ):
                print( "Incorrect mask dimensions")
                self._mask_image_data = None
        else:
            self._mask_image_data = None
        
        self._display_current_frame()
        self._plot_canvas.canvas.mpl_connect('pick_event', self._on_pick)
        if( not self.is_window_showing ):
            self._plot_canvas.canvas.mpl_connect('close_event', self._handle_close)
            plt.ion()
            plt.show()
            self.is_window_showing = True
            
    def set_mask_transparency( self, newTransparency ):
        _mask_transparency = newTransparency
        self._display_current_frame()
        
    def set_mask_showing( self, showing ):
        self._is_mask_displayed = showing
        self._display_current_frame()
            
    def _handle_close(self, evt):
        self.is_window_showing = False
        plt.close('all')
        
    def _display_current_frame( self ):
        if( self._plot_canvas == None ):
            self._plot_canvas, self._plot_axes = plt.subplots( nrows = 1, ncols = 3 )
                                
        slices = [ self._base_image_data[ self._axial_pos, : , :     ],
                   self._base_image_data[ : , self._saggital_pos , : ],
                   self._base_image_data[ : , : , self._coronal_pos  ] ] 
        self._plot_artists = [ None, None, None ]
        
        if( self._is_mask_displayed and not self._mask_image_data is None ):
            mask_slices = [ self._mask_image_data[ self._axial_pos, : , :     ],
                            self._mask_image_data[ : , self._saggital_pos , : ],
                            self._mask_image_data[ : , : , self._coronal_pos  ] ] 

        
        for i, slice in enumerate(slices):
            self._plot_axes[ i ].clear()
            self._plot_artists[ i ] = self._plot_axes[ i ].imshow( slice.T,         \
                                                                   cmap="gray",     \
                                                                   origin="lower",  \
                                                                   norm = Normalize(vmax=self._image_min_max[1], vmin=self._image_min_max[0]),     \
                                                                   picker = True )
            if( self._is_mask_displayed and not self._mask_image_data is None ):
                self._plot_axes[ i ].imshow( mask_slices[ i ].T,      \
                                             cmap="viridis",          \
                                             origin="lower",          \
                                             alpha = self._mask_transparency,    \
                                             picker = False )
            
        self._plot_canvas.canvas.draw()
        
    def _move_slice( self, axial_delta, saggital_delta, coronal_delta ):
        self._axial_pos = self._axial_pos + axial_delta
        self._axial_pos = self._axial_pos % self._base_image_data.shape[ 0 ]
        
        self._saggital_pos = self._saggital_pos + saggital_delta
        self._saggital_pos = self._saggital_pos % self._base_image_data.shape[ 1 ]
        
        self._coronal_pos = self._coronal_pos + coronal_delta
        self._coronal_pos = self._coronal_pos % self._base_image_data.shape[ 2 ]
       
    def _on_pick(self, event):
        mouseevent = event.mouseevent
        artist = event.artist
        delta = 0
        if mouseevent.button == 'up':
            delta = 2
            print( "scroll up")
        elif mouseevent.button == 'down':
            delta = -2
            print( "scroll down")
        if delta != 0 and isinstance(artist, AxesImage):
            #Axial plot
            if artist == self._plot_artists[ 0 ]:
                print( "Pick on axial image")
                self._move_slice( delta, 0, 0 )
        
            #Saggital plot
            if artist == self._plot_artists[ 1 ]:
                print( "Pick on saggital image")
                self._move_slice( 0, delta, 0 )
                
            #Coronal plot
            if artist == self._plot_artists[ 2 ]:
                print( "Pick on coronal image")
                self._move_slice( 0, 0, delta )
                
            self._display_current_frame()
            
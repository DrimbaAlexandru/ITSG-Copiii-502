import tkinter as tk
import nibabel as nib
from niiPlot import MRI_plot 
import unet.main as main
import numpy as np
from tqdm import tqdm
from skimage.io import  imsave
from skimage.transform import resize
from skimage.color import gray2rgb

class App:
    IMG_CHANNELS = 3
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    RESULTS_PATH = "./output/"
    #Class variables
    root = None
    image_path = ""
    label_path = ""
    plot_canvas = None
    
    _mask_enabled_var = None
    _slider = None
    model = None

    #Procedures
    def __init__(self, master):       
        print( "init")
        #Initialize variables
        self.root = master
        
        self.root.geometry("200x200")
        self.root.resizable(0, 0)
        
        #Class constants
        self._menu_file = [( "Open NIfTI image...",         self._on_load_image         ), 
                           ( "Open NIfTI image labels...",  self._on_load_labels        ), 
                           ( "Separator",                   None                        ), 
                           ( "Generate labels...",          self._not_yet_implemented   ),
                           ( "Separator",                   None                        ), 
                           ( "Exit",                        self.root.quit              )]

        self._menu_options = [( "ML method", self._not_yet_implemented )]

        self._menus = [ ( "File", self._menu_file ), ( "Options", self._menu_options ) ]

        self.root.title("The best project in the whole goddamn world")
        
        #Initialize the menu bar
        self._init_menus()
        
        self._mask_enabled_var = tk.IntVar()
        self._mask_enabled_var.set( 1 )
        c = tk.Checkbutton( self.root, text="Show image masks", variable=self._mask_enabled_var, command=self._on_dispay_mask_changed )
        c.pack()
        self.model = main.load_model()
        
        # self._slider = Scale(self.root, from_=0, to=1, orient=HORIZONTAL, command = self._on_slider_moved)
        # self._slider.pack()

    def _init_menus( self ):
        # create a toplevel menu
        self.menubar = tk.Menu(self.root)
        
        for menuName, options in self._menus:
            filemenu = tk.Menu( self.menubar, tearoff = 0 )
            for subMenuName, action in options:
                if action is not None:
                    filemenu.add_command( label = subMenuName, command = action )
                else:
                    filemenu.add_separator()
            self.menubar.add_cascade( label = menuName, menu = filemenu )
        
        # display the menu
        self.root.config( menu = self.menubar )
        
    def _on_load_image( self ):
        self.image_path = tk.filedialog.askopenfilename(parent=self.root, initialdir = "/",title = "Select image file",filetypes = (("NIfTI files","*.nii.gz;*.nii"),("all files","*.*")))
        print(self.image_path)
        self.label_path = ""
        if( self.image_path != "" ):
            self._display_nifti_image()
            self.generate_mask()

    def generate_mask( self ):
        print("Generate mask");

        proxy_img = nib.load( self.image_path )
        canonical_img = nib.as_closest_canonical(proxy_img)
        image_data = canonical_img.get_fdata()
        min_max = ( image_data.min(), image_data.max() )

        numslices0 = image_data.shape[ 0 ]
        numslices1 = image_data.shape[ 1 ]
        numslices2 = image_data.shape[ 2 ]

        print(image_data.ndim)
        print(image_data.shape)
        print(image_data.size)
           
        name = 'result.nii.gz'

        result  = image_data
        new_header = header = proxy_img.header.copy()
        new_data = np.zeros( ( self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_WIDTH ), dtype=np.uint8 )
        new_img = nib.nifti1.Nifti1Image(new_data, None, header=new_header)
        print(new_data.ndim)
        print(new_data.shape)
        print(new_data.size)
        for i in tqdm( range(numslices0) ):
            img = image_data[i,:,:]
            img = np.flip( img.T, axis = 0 )
            if len( img.shape ) == 2 or ( self.IMG_CHANNELS == 3 and img.shape[2] != self.IMG_CHANNELS ):
                img = gray2rgb( img )
            img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True )
            img = img * ( 255 / min_max[1] )
            img = img.astype(np.uint8)
            result = self.model.predict_one_from_memory(img)
            
            print(result.ndim)
            print(result.shape)
            print(result.size)
                
            new_data[i,:,:] = result

        imsave( self.RESULTS_PATH + name, result)
        
    def _on_load_labels( self ):
        self.label_path = tk.filedialog.askopenfilename(parent=self.root, initialdir = "/",title = "Select image mask",filetypes = (("NIfTI files","*.nii.gz;*.nii"),("all files","*.*")))
        if( self.label_path != "" ):
            self._display_nifti_image()
        
    def _not_yet_implemented( self ):
        print( "castraveCiori")
        
    def _display_nifti_image( self ):       
        if( self.plot_canvas is None or ( not self.plot_canvas.is_window_showing)):
            self.plot_canvas = MRI_plot( self.image_path, self.label_path )
        else:
            self.plot_canvas.set_image_paths( self.image_path, self.label_path )

    def _on_dispay_mask_changed( self ):
        if( self.plot_canvas is not None ):
            if( self._mask_enabled_var.get() == 0 ):
                self.plot_canvas.set_mask_showing( False )
            else:
                self.plot_canvas.set_mask_showing( True )
                    
    def _on_slider_moved( self, evnt ):
        if( self.plot_canvas is not None ):
            self.plot_canvas.set_mask_transparency( self._slider.get() )

        
#Create and start de app
root = tk.Tk()

app = App(root)

root.mainloop()
#root.destroy() # optional; see description below
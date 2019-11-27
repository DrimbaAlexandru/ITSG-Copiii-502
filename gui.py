import os

import tkinter as tk
import nibabel as nib
from niiPlot import MRI_plot 
from unet.unetModel import Unet_model


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
                           ( "Generate labels...",          self._generate_mask   ),
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
        self.model = Unet_model( 3,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 [ ( ( 0, 0, 0 ), "Background" ), ( ( 127, 127, 127 ), "Ventricular Myocardum" ), ( ( 255, 255, 255 ), "Blood Pool" ) ] )
        self.model.load_model()

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

    def _generate_mask( self ):
        print("Generate mask");

        if self.image_path == "" :
            return

        proxy_img = nib.load( self.image_path )
        canonical_img = nib.as_closest_canonical(proxy_img)
        image_data = canonical_img.get_fdata()

        generated_mask = self.model.predict_volume( image_data )

        name = 'result.nii.gz'
        os.makedirs(self.RESULTS_PATH,exist_ok=True)

        new_header = proxy_img.header.copy()
        new_img = nib.nifti1.Nifti1Image(generated_mask, None, header=new_header)
        print(generated_mask.ndim)
        print(generated_mask.shape)
        print(generated_mask.size)

        img = nib.Nifti1Image(generated_mask, canonical_img.affine)
        img.to_filename(self.RESULTS_PATH + name)
        nib.save(img, self.RESULTS_PATH + name)


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
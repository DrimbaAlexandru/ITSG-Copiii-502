import os

import tkinter as tk
from niiPlot import MRI_plot
from unet.unet3DModelWithGenerator import Unet3DModelWithGenerator

import utils.utils as myUtils

class App:
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
                           ( "Generate labels...",          self._generate_mask         ),
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
        
        self._slider = tk.Scale(self.root, from_=0, to=1, orient=tk.HORIZONTAL, resolution=0.05, command = self._on_slider_moved)
        self._slider.pack()
        
        self._init_model()

    def _init_model( self ):
        IMG_SIZE = 64
        self.model = Unet3DModelWithGenerator([ ( (0), "Background" ), ( (127), "Ventricular Myocardum" ), ( (255), "Blood Pool" ) ],
                                              IMG_SIZE)
        self.model.load_model()

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
        print("Generate mask")

        if self.image_path == "" :
            return

        image_data, affine = myUtils.load_nifti_image( self.image_path )

        generated_mask = self.model.predict_volume( image_data )

        name = 'result.nii.gz'
        os.makedirs(self.RESULTS_PATH,exist_ok=True)
        
        myUtils.save_nifti_image(generated_mask, affine, self.RESULTS_PATH + name)

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
root.destroy() # optional; see description below

#Pentru mindmap: chestii tehnice, sau chestii din procesul de dezvoltare. (idee->date->algoritm etc. )

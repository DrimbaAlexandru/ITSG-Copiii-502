from Tkinter import *
import tkFileDialog

from niiPlot import MRI_plot 

class App:
    
    #Class variables
    root = None
    image_path = None
    label_path = None
    plot_canvas = None
    
    #Procedures
    def __init__(self, master):       
        print "init"
        #Initialize variables
        self.root = master
        
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

    def _init_menus( self ):
        # create a toplevel menu
        self.menubar = Menu(self.root)
        
        for menuName, options in self._menus:
            filemenu = Menu( self.menubar, tearoff = 0 )
            for subMenuName, action in options:
                if action is not None:
                    filemenu.add_command( label = subMenuName, command = action )
                else:
                    filemenu.add_separator()
            self.menubar.add_cascade( label = menuName, menu = filemenu )
        
        # display the menu
        self.root.config( menu = self.menubar )
        
    def _on_load_image( self ):
        self.image_path = tkFileDialog.askopenfilename(parent=self.root, initialdir = "/",title = "Select image file",filetypes = (("NIfTI files","*.nii.gz;*.nii"),("all files","*.*")))
        self.label_path = None
        if( self.image_path is not None ):
            self._display_nifti_image()
        
    def _on_load_labels( self ):
        self.label_path = tkFileDialog.askopenfilename(parent=self.root, initialdir = "/",title = "Select image mask",filetypes = (("NIfTI files","*.nii.gz;*.nii"),("all files","*.*")))
        if( self.label_path is not None ):
            self._display_nifti_image()
        
    def _not_yet_implemented( self ):
        print "castraveCiori"
        
    def _display_nifti_image( self ):       
        if( self.plot_canvas is None or ( not self.plot_canvas.is_showing)):
            self.plot_canvas = MRI_plot( self.image_path, self.label_path )
        else:
            self.plot_canvas.set_image_paths( self.image_path, self.label_path )

        
#Create and start de app
root = Tk()

app = App(root)

root.mainloop()
root.destroy() # optional; see description below
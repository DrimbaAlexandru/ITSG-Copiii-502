from Tkinter import *
import tkFileDialog

class App:
    
    #Class variables
    root = None
    image_path = None
    label_path = None
    plot_canvas = None
    
    #Procedures
    def __init__(self, master):        
        #Initialize variables
        self.root = master
        
        #Class constants
        self._menu_file = [ ( "Open NIfTI image...",         self._on_load_image         ), 
               ( "Open NIfTI image labels...",  self._on_load_labels        ), 
               ( "Separator",                   None                        ), 
               ( "Generate labels...",          self._not_yet_implemented   ),
               ( "Separator",                   None                        ), 
               ( "Exit",                        self.root.quit              ) ]

        self._menu_options = [ ( "ML method",                self._not_yet_implemented   ) ]

        self._menus = [ ( "File", self._menu_file ), ( "Options", self._menu_options ) ]

        #Initialize the menu bar
        self._init_menus()

    def _init_menus( self ):
        # create a toplevel menu
        self.menubar = Menu(self.root)
        
        for menuName, options in self._menus:
            filemenu = Menu( self.menubar, tearoff = 0 )
            for subMenuName, action in options:
                if action != None:
                    filemenu.add_command( label = subMenuName, command = action )
                else:
                    filemenu.add_separator()
            self.menubar.add_cascade( label = menuName, menu = filemenu )
        
        # display the menu
        self.root.config( menu = self.menubar )
        
    def _on_load_image( self ):
        self.image_path = tkFileDialog.askopenfilename(initialdir = "/",title = "Select image file",filetypes = (("NIfTI files","*.nii.gz;*.nii"),("all files","*.*")))
        self._display_nifti_image()
        
    def _on_load_labels( self ):
        self.label_path = tkFileDialog.askopenfilename(initialdir = "/",title = "Select image mask",filetypes = (("NIfTI files","*.nii.gz;*.nii"),("all files","*.*")))
        self._display_nifti_image()
        
    def _not_yet_implemented( self ):
        print "castraveCiori"
        
    def _display_nifti_image( self ):
        print self.image_path
        print self.label_path

        
#Create and start de app
root = Tk()

app = App(root)

root.mainloop()
root.destroy() # optional; see description below
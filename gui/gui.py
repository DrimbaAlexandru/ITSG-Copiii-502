import tkinter as tk
import webbrowser

from gui.components.controlsGUIComponent import ControlsComponent
from gui.components.niftiPlotGUIComponent import MRIPlotComponent
from service.app_service import AppService


class AppGUI:
    IMG_CHANNELS = 3
    # Class variables
    root = None
    image_path = ""
    label_path = ""
    plot_canvas = None

    _slider = None

    # Procedures
    def __init__(self, master):
        # Initialize variables
        self.root = master
        self.root.geometry("800x500")
        self.root.resizable(1, 1)
        self.root.title("The best project in the whole goddamn world")

        # Class constants
        self._menu_file = [("Open NIfTI image", self._on_load_image),
                           ("Open NIfTI image labels", self._on_load_labels),
                           ("Separator", None),
                           ("Exit", self.root.quit)]
        self._menu_options = [("ML method", self._on_change_ML_method)]
        self._menus = [("File", self._menu_file), ("Options", self._menu_options)]

        # Initialize the menu bar
        self._init_menus()
        # self._init_window()
        self._service = AppService()
        self._controls = ControlsComponent(self.root, self._service)
        self.plot_canvas = MRIPlotComponent(self.root, self._service)
        self._controls.set_plot_canvas(self.plot_canvas)

        webbrowser.register(name='chrome', klass=webbrowser.Chrome('chrome'))

    def _init_menus(self):
        # create a toplevel menu
        self.menubar = tk.Menu(self.root)

        for menuName, options in self._menus:
            file_menu = tk.Menu(self.menubar, tearoff=0)
            for subMenuName, action in options:
                if action is not None:
                    file_menu.add_command(label=subMenuName, command=action)
                else:
                    file_menu.add_separator()
            self.menubar.add_cascade(label=menuName, menu=file_menu)

        # display the menu
        self.root.config(menu=self.menubar)

    def _on_load_image(self):
        self.image_path = tk.filedialog.askopenfilename(parent=self.root, initialdir="/", title="Select image file",
                                                        filetypes=(
                                                            ("NIfTI files", "*.nii.gz;*.nii"), ("all files", "*.*")))
        self._service.set_image_path(self.image_path)
        self.label_path = ""
        if self.image_path != "":
            self._display_nifti_image()

    def _on_load_labels(self):
        self.label_path = tk.filedialog.askopenfilename(parent=self.root, initialdir="/", title="Select image mask",
                                                        filetypes=(
                                                            ("NIfTI files", "*.nii.gz;*.nii"), ("all files", "*.*")))
        self._service.set_labels_path(self.label_path)
        if self.label_path != "":
            self._display_nifti_image()

    def _on_change_ML_method(self):
        print("Not yet implemented")

    def _display_nifti_image(self):
        self.plot_canvas.set_image_paths(self.image_path, self.label_path)

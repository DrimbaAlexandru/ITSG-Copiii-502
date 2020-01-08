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
    unet_3d = None

    _slider = None

    # Procedures
    def __init__(self, master):
        # Initialize variables
        self.root = master
        self.root.geometry("800x500")
        self.root.resizable(1, 1)
        self.root.title("The best project in the whole goddamn world")

        self.unet_3d = tk.BooleanVar(master)
        self.unet_3d.set(True)

        self.previous_unet_3d = True

        # Class constants
        self._menu_file = [("Open NIfTI image", self._on_load_image),
                           ("Open NIfTI image labels", self._on_load_labels),
                           ("Separator", None),
                           ("Exit", self.root.quit)]
        self._menus = [("File", self._menu_file)]

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

        self._init_ml_method_menu()

        # display the menu
        self.root.config(menu=self.menubar)

    def _init_ml_method_menu(self):
        method_menu = tk.Menu(self.menubar, tearoff=0)

        method_menu.add_radiobutton(label="3D U-net model", value=1, variable=self.unet_3d,
                                    command=self._change_to_3d_model)
        method_menu.add_radiobutton(label="2D U-net model", value=0, variable=self.unet_3d,
                                    command=self._change_to_2d_model)

        self.menubar.add_cascade(label="ML Method", menu=method_menu)

    def _change_to_2d_model(self):
        if self.previous_unet_3d:
            self._service.set_2d_model()
            self.previous_unet_3d = False

    def _change_to_3d_model(self):
        if not self.previous_unet_3d:
            self._service.set_3d_model()
            self.previous_unet_3d = True

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

    def _display_nifti_image(self):
        self.plot_canvas.set_image_paths(self.image_path, self.label_path)

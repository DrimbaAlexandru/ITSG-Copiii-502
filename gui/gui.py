import os
import tkinter as tk
import webbrowser

import utils.utils as utils
from gui.components.controlsGUIComponent import ControlsComponent
from gui.components.niftiPlotGUIComponent import MRIPlotComponent
from service.app_service import AppService


class AppGUI:
    IMG_CHANNELS = 3

    # Procedures
    def __init__(self, master):
        # Initialize variables
        self.root = master
        self.root.geometry("800x500")
        self.root.resizable(1, 1)
        self.root.title("The best project in the whole goddamn world")

        self._ml_methods = [ ("3D U-net model", self._set_3D_Unet_model),
                             ("2D U-net model", self._set_2D_Unet_model)]
        self._ml_method_idx = tk.IntVar(master)
        self._ml_method_idx.set(0)
        self._loaded_ml_method_idx = -1 # Init with invalid idx, so that any new index loads the ML model

        # Class constants
        self._menu_file = [("Open NIfTI image", self._on_load_image),
                           ("Open NIfTI image labels", self._on_load_labels),
                           ("Separator", None),
                           ("Render 3D labels", self._on_render_3d),
                           ("Separator", None),
                           ("Exit", self.root.quit)]
        self._menus = [("File", self._menu_file)]

        # Initialize the menu bar
        self._init_menus()
        # self._init_window()
        self._service = AppService()
        self._set_model()
        self._controls = ControlsComponent(self.root, self._service)
        self.plot_canvas = MRIPlotComponent(self.root, self._service)
        self._controls.set_plot_canvas(self.plot_canvas)

        self.image_path = ""
        self.label_path = ""

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

        for idx, (model_name, _) in enumerate(self._ml_methods):
            method_menu.add_radiobutton(label=model_name, value=idx, variable=self._ml_method_idx,
                                        command=self._set_model)

        self.menubar.add_cascade(label="ML Method", menu=method_menu)

    def _set_model(self):
        if self._loaded_ml_method_idx != self._ml_method_idx.get():
            self._ml_methods[self._ml_method_idx.get()][1]()
            self._loaded_ml_method_idx = self._ml_method_idx.get()

    def _set_2D_Unet_model(self):
        self._service.set_2d_model()

    def _set_3D_Unet_model(self):
        self._service.set_3d_model()

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

    def _on_render_3d(self):
        label_path = tk.filedialog.askopenfilename(parent=self.root, initialdir="/", title="Select image mask",
                                                   filetypes=(
                                                       ("NIfTI files", "*.nii.gz;*.nii"), ("all files", "*.*")))
        self._service.generate_3d_rendered(label_path, utils.get_path_for_saving, self._on_3d_mask_generated)

    def _on_3d_mask_generated(self, path):
        if path != "":
            url = os.path.abspath(path)
            webbrowser.open(url, new=2)
        else:
            print("WARN: The path for the 3d rendered mask is not valid")

    def _display_nifti_image(self):
        self.plot_canvas.set_image_paths(self.image_path, self.label_path)

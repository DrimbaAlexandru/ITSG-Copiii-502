import os
import tkinter as tk
import webbrowser

from service.app_service import AppService


class ControlsComponent:

    def __init__(self, root, app_service: AppService):
        self._root = root
        self._app_service = app_service
        self._plot_canvas = None

        self._labels_generated_text = tk.StringVar(value="")
        self._3d_mask_generated_text = tk.StringVar(value="")

        self._mask_enabled_var = tk.IntVar()
        self._mask_enabled_var.set(0)

        self._init_first_row()
        self._init_second_row()
        self._init_third_row()

        app_service.subscribe_to_image_changes(self._on_image_change_callback)

    def set_plot_canvas(self, plot_canvas):
        self._plot_canvas = plot_canvas

    def _init_first_row(self):
        container = tk.Frame(self._root)

        button = tk.Button(self._root, text="Generate labels", command=self._on_generate_labels)
        button.pack(in_=container, side="left")

        label = tk.Label(self._root, textvariable=self._labels_generated_text)
        label.pack(in_=container, side="left")

        container.pack()

    def _init_second_row(self):
        container = tk.Frame(self._root)

        c = tk.Checkbutton(self._root, text="Show image masks", variable=self._mask_enabled_var,
                           command=self._on_display_mask_changed)
        c.pack(in_=container, side="left")

        self._slider = tk.Scale(self._root, from_=0, to=1, orient=tk.HORIZONTAL, resolution=0.05,
                                command=self._on_slider_moved)
        self._slider.pack(in_=container, side="left")

        container.pack()

    def _init_third_row(self):
        container = tk.Frame(self._root)

        button = tk.Button(self._root, text="Open 3D rendered masks", command=self._on_open_3d_rendered)
        button.pack(in_=container, side="left")

        label = tk.Label(self._root, textvariable=self._3d_mask_generated_text)
        label.pack(in_=container, side="left")

        container.pack()

    def _on_generate_labels(self):
        self._app_service.generate_mask(self._get_path_for_saving, self._on_labels_generated,
                                        self._on_3d_mask_generated)

    def _on_display_mask_changed(self):
        if self._plot_canvas is not None:
            if self._mask_enabled_var.get() == 0:
                self._plot_canvas.set_mask_showing(False)
            else:
                self._plot_canvas.set_mask_showing(True)

    def _on_slider_moved(self, event):
        if self._plot_canvas is not None:
            self._plot_canvas.set_mask_transparency(self._slider.get())

    def _on_open_3d_rendered(self):
        path = self._app_service.get_3d_mask_path()
        if path != "":
            url = os.path.abspath(path)
            webbrowser.open(url, new=2)
        else:
            print("WARN: The path for the 3d rendered mask is not valid anymore")

    def _on_labels_generated(self, path):
        self._labels_generated_text.set("Labels generated at " + path)
        self._mask_enabled_var.set(1)
        self._on_display_mask_changed()

    def _on_3d_mask_generated(self, path):
        self._3d_mask_generated_text.set("3D mask generated at " + path)

    def _on_image_change_callback(self, path_labels, path_3d_mask):
        if path_labels is not None and path_labels != "":
            self._labels_generated_text.set("Labels generated at " + path_labels)
        else:
            self._labels_generated_text.set("")

        if path_3d_mask is not None and path_3d_mask != "":
            self._3d_mask_generated_text.set("3D mask generated at " + path_3d_mask)
        else:
            self._3d_mask_generated_text.set("")

    @staticmethod
    def _get_path_for_saving(extension):
        f = tk.filedialog.asksaveasfile(mode='w', defaultextension=extension)
        if f is None:
            return None
        else:
            return f.name

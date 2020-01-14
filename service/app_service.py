import nibabel as nib

import utils.utils as myUtils
from renderer.renderer3d import Renderer3D
from unet.unet_2D.unet2DModel import Unet2DModel
from unet.unet_3D.unet3DModelWithGenerator import Unet3DModelWithGenerator


class AppService:
    def __init__(self):
        self._image_path = ""
        self._label_path = ""
        self._3d_mask_path = ""
        self._model = None
        self._renderer = Renderer3D()
        self._on_image_change_callback = None

    def set_image_path(self, image_path):
        self._image_path = image_path

        # when loading a new nifti image, the generated labels are not corresponding anymore
        self._label_path = ""
        self._3d_mask_path = ""

        if self._on_image_change_callback is not None:
            self._on_image_change_callback(self._label_path, self._3d_mask_path)

    def get_image_path(self):
        return self._image_path

    def set_3d_model(self):
        print("INFO: Set 3D U-Net model")
        self._model = Unet3DModelWithGenerator([((0), "Background"), ((127), "Ventricular Myocardum"),
                                                ((255), "Blood Pool")],
                                               64)
        self._model.load_model()
        print("INFO: 3D U-Net model was successfully set")

    def set_2d_model(self):
        print("INFO: Set 2D U-Net model")
        self._model = Unet2DModel([((0, 0, 0), "Background"), ((127, 127, 127), "Ventricular Myocardum"),
                                   ((255, 255, 255), "Blood Pool")],
                                  128,
                                  3)
        self._model.load_model()
        print("INFO: 2D U-Net model was successfully set")

    def generate_mask(self, get_path_for_saving_callback, labels_generated_callback, rendered_mask_generated_callback):
        if self._image_path == "":
            print("ERROR: There is no nifti image opened. Cannot generate labels!")
            return

        print("INFO: Generate labels")

        image_data, affine = myUtils.load_nifti_image(self._image_path)

        generated_mask = self._model.predict_volume(image_data)
        self._save_main_mask(generated_mask, affine, get_path_for_saving_callback, labels_generated_callback)

        iframe_to_save = self._renderer.generate(generated_mask)
        self._save_3d_rendered_mask(iframe_to_save, get_path_for_saving_callback, rendered_mask_generated_callback)

    def generate_3d_rendered(self, image_path, get_path_for_saving_callback, rendered_mask_generated_callback):
        # nifti_image = nib.load(image_path)
        image_data, affine = myUtils.load_nifti_image(image_path)
        iframe_to_save = self._renderer.generate(image_data)
        self._save_3d_rendered_mask(iframe_to_save, get_path_for_saving_callback, rendered_mask_generated_callback)

    def _save_main_mask(self, image_to_save, affine, get_path_for_saving_callback, labels_generated_callback):
        path = get_path_for_saving_callback(".nii")
        if path is None:
            print("WARN: Labels were not saved due to no path or invalid path provided! Please generate again!")
            return
        try:
            myUtils.save_nifti_image(image_to_save, affine, path)
            self._label_path = path
            labels_generated_callback(path)
            print("INFO: Labels were successfully generated and saved!")
        except Exception:
            print("ERROR: Labels could not be saved! Please generate again!")

    def _save_3d_rendered_mask(self, iframe_to_save, get_path_for_saving_callback, rendered_mask_generated_callback):
        path = get_path_for_saving_callback(".html")
        if path is None:
            print(
                "WARN: 3d rendered mask was not saved due to no path or invalid path provided! Please generate again!")
            return
        iframe_to_save.write_html(path)
        self._3d_mask_path = path
        rendered_mask_generated_callback(path)
        print("INFO: 3d rendered mask was successfully generated and saved!")

    def subscribe_to_image_changes(self, callback):
        self._on_image_change_callback = callback

    def get_label_path(self):
        return self._label_path

    def set_labels_path(self, path):
        self._label_path = path
        if self._on_image_change_callback is not None:
            self._on_image_change_callback(self._label_path, self._3d_mask_path)

    def get_3d_mask_path(self):
        return self._3d_mask_path

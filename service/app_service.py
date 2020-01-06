import nibabel as nib

from unet.unetModel import UnetModel
from renderer.renderer3d import Renderer3D


class AppService:
    def __init__(self):
        self._image_path = ""
        self._label_path = ""
        self._3d_mask_path = ""
        self._model = None
        self._renderer = Renderer3D()
        self._on_image_change_callback = None

        self._init_model()

    def set_image_path(self, image_path):
        self._image_path = image_path

        # when loading a new nifti image, the generated labels are not corresponding anymore
        self._label_path = ""
        self._3d_mask_path = ""

        if self._on_image_change_callback is not None:
            self._on_image_change_callback(self._label_path, self._3d_mask_path)

    def get_image_path(self):
        return self._image_path

    def _init_model(self):
        self._model = UnetModel(3,
                                None,
                                None,
                                None,
                                None,
                                None,
                                [((0, 0, 0), "Background"), ((127, 127, 127), "Ventricular Myocardum"),
                                 ((255, 255, 255), "Blood Pool")])
        self._model.load_model()

    def generate_mask(self, get_path_for_saving_callback, labels_generated_callback, rendered_mask_generated_callback):
        if self._image_path == "":
            print("ERROR: There is no nifti image opened. Cannot generate labels!")
            return

        print("INFO: Generate labels")

        proxy_img = nib.load(self._image_path)
        canonical_img = nib.as_closest_canonical(proxy_img)
        image_data = canonical_img.get_fdata()

        generated_mask = self._model.predict_volume(image_data)

        image_to_save = nib.Nifti1Image(generated_mask, canonical_img.affine)

        self._save_main_mask(image_to_save, get_path_for_saving_callback, labels_generated_callback)
        self._save_3d_rendered_mask(image_to_save, get_path_for_saving_callback, rendered_mask_generated_callback)

    def _save_main_mask(self, image_to_save, get_path_for_saving_callback, labels_generated_callback):
        path = get_path_for_saving_callback(".nii")
        if path is None:
            print("WARN: Labels were not saved due to no path or invalid path provided! Please generate again!")
            return
        try:
            self.save_nifti_image(image_to_save, path)
            self._label_path = path
            labels_generated_callback(path)
            print("INFO: Labels were successfully generated and saved!")
        except Exception:
            print("ERROR: Labels could not be saved! Please generate again!")

    def _save_3d_rendered_mask(self, image, get_path_for_saving_callback, rendered_mask_generated_callback):
        iframe_to_save = self._renderer.generate(image)
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

    @staticmethod
    def save_nifti_image(image, path):
        nib.save(image, path)

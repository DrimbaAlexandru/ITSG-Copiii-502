from renderer3d import Renderer3D
import nibabel as nib

TEST_IMAGE_PATH = '.././data/Axialcropped/truth/training_axial_crop_pat3-label.nii'
nifti_image = nib.load(TEST_IMAGE_PATH)

preprocessor = Renderer3D()
preprocessor.plot(nifti_image)

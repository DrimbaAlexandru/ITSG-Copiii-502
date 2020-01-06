import nibabel as nib
from renderer.renderer3d import Renderer3D

TEST_IMAGE_PATH = '.././output/result.nii.gz'
nifti_image = nib.load(TEST_IMAGE_PATH)

preprocessor = Renderer3D()
preprocessor.plot(nifti_image)

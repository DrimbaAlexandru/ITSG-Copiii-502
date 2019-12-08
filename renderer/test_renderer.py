from renderer3d import Renderer3D
import nibabel as nib

TEST_IMAGE_PATH = 'D:/git/ITSG-Copiii-502/output/result.nii.gz'
nifti_image = nib.load(TEST_IMAGE_PATH)

preprocessor = Renderer3D()
preprocessor.plot(nifti_image)

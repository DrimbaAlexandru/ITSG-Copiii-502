import matplotlib.pyplot as plt
import nibabel as nib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


class OldRenderer3D:
    def plot(self, nifti_image):
        print("Plot main function...")
        label_image_data = self._process_nifti(nifti_image)
        self._plot_3d(label_image_data)

    def _plot_3d(self, label_image_data):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        mesh1 = self._generate_mesh(label_image_data, level=0.5, mesh_alpha=0.3, mesh_color=[0.45, 0.45, 0.75])
        ax.add_collection3d(mesh1)

        mesh2 = self._generate_mesh(label_image_data, level=1, mesh_alpha=0.7, mesh_color=[1, 0.27, 0.27])
        ax.add_collection3d(mesh2)

        ax.set_xlim(0, label_image_data.shape[0])
        ax.set_ylim(0, label_image_data.shape[1])
        ax.set_zlim(0, label_image_data.shape[2])

        plt.show()

    @staticmethod
    def _generate_mesh(label_image_data, level, mesh_alpha, mesh_color):
        verts, faces, norm, val = measure.marching_cubes_lewiner(label_image_data, level, step_size=1,
                                                                 allow_degenerate=True)
        mesh = Poly3DCollection(verts[faces], alpha=mesh_alpha)
        mesh.set_facecolor(mesh_color)
        return mesh

    @staticmethod
    def _process_nifti(nifti_image):
        canonical_img = nib.as_closest_canonical(nifti_image)
        image_data = canonical_img.get_fdata()
        return image_data

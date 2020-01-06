import plotly.graph_objects as go
from skimage import measure


class Renderer3D:
    def generate(self, image):
        print("INFO: Start generating 3d rendered mask...")
        return self._plot_3d(image)

    def _plot_3d(self, label_image_data):
        mesh1 = self._generate_mesh(label_image_data, level=0.5, mesh_alpha=0.3, mesh_color='lightpink')
        mesh2 = self._generate_mesh(label_image_data, level=1, mesh_alpha=0.5, mesh_color='cyan')
        fig = go.Figure(data=[mesh2, mesh1])
        print("INFO: Finished generating 3d mask")
        return fig

    @staticmethod
    def _generate_mesh(label_image_data, level, mesh_alpha, mesh_color):
        verts, faces, norm, val = measure.marching_cubes_lewiner(label_image_data, level, step_size=1,
                                                                 allow_degenerate=True)
        mesh = go.Mesh3d(x=[vert[0] for vert in verts],
                         y=[vert[1] for vert in verts],
                         z=[vert[2] for vert in verts],
                         color=mesh_color,
                         i=[face[0] for face in faces],
                         j=[face[1] for face in faces],
                         k=[face[2] for face in faces],
                         name='y',
                         showscale=True,
                         opacity=mesh_alpha)
        return mesh

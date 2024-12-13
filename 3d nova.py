import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.io import load_objs_as_meshes
import matplotlib.pyplot as plt

def render_3d_model():
    # Create a simple cube mesh for demonstration
    verts = torch.tensor([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                          [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]], dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6],
                          [0, 2, 4], [2, 6, 4], [1, 3, 5], [3, 7, 5],
                          [0, 1, 4], [1, 5, 4], [2, 3, 6], [3, 7, 6]], dtype=torch.int64)

    textures = TexturesVertex(verts_features=torch.ones_like(verts)[None])

    # Construct mesh
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

    # Set up renderer
    R, T = look_at_view_transform(2.7, 0, 180)  # Distance, elevation, azimuth
    cameras = FoVPerspectiveCameras(device="cpu", R=R, T=T)
    raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
    lights = PointLights(device="cpu", location=[[0.0, 0.0, -3.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device="cpu", cameras=cameras, lights=lights)
    )

    # Render the mesh
    image = renderer(mesh)

    # Display the rendered image
    plt.imshow(image[0, ..., :3].detach().cpu().numpy())
    plt.axis("off")
    plt.show()

# Run the rendering function
render_3d_model()

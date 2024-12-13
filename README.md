# 3D Cube Rendering with PyTorch3D

## Project Overview
This project demonstrates how to render a simple 3D cube using PyTorch3D, a library for 3D deep learning research. The project sets up a rendering pipeline to visualize a 3D model using basic rendering techniques such as rasterization and shading.

## Features
- Renders a 3D cube model using PyTorch3D.
- Implements basic Phong shading for realistic lighting effects.
- Configures a camera to provide a perspective view of the model.

## Prerequisites
To run this project, you need:
- Python 3.8 or higher
- PyTorch (compatible with your system's CUDA version, if applicable)
- PyTorch3D
- Matplotlib

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/3d-cube-rendering.git
   cd 3d-cube-rendering
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the required Python packages:
   ```bash
   pip install torch torchvision
   pip install pytorch3d
   pip install matplotlib
   ```

## Usage

1. Run the script:
   ```bash
   python render_cube.py
   ```

2. The script will render a 3D cube and display the result in a Matplotlib window.

## Code Explanation

- **Vertices and Faces:**
  The cube is defined by its vertices and triangular faces.
  ```python
  verts = torch.tensor([...])
  faces = torch.tensor([...])
  ```

- **Textures:**
  A uniform texture is applied to the cube.
  ```python
  textures = TexturesVertex(verts_features=torch.ones_like(verts)[None])
  ```

- **Renderer:**
  The renderer uses a perspective camera, rasterizer, and a Phong shader.
  ```python
  renderer = MeshRenderer(
      rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
      shader=SoftPhongShader(device="cpu", cameras=cameras, lights=lights)
  )
  ```

- **Rendering:**
  The rendered image is displayed using Matplotlib.
  ```python
  plt.imshow(image[0, ..., :3].detach().cpu().numpy())
  ```

## Output
The output is a rendered image of a 3D cube displayed in a Matplotlib window. The cube is shaded with Phong lighting and can be adjusted by modifying camera and light settings in the script.

## Customization
- **Camera Settings:** Modify the `look_at_view_transform` parameters to change the viewing angle.
- **Lighting:** Adjust the light location in `PointLights` for different illumination effects.
- **Mesh:** Replace the cube vertices and faces with a different 3D model.

## Troubleshooting
- Ensure PyTorch3D is installed correctly. Refer to the [official installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for system-specific instructions.
- If the rendering is slow or crashes, check your GPU memory or use a smaller `image_size` in `RasterizationSettings`.

## License
This project is licensed under the MIT License. Feel free to use and modify the code.

## Acknowledgements
- [PyTorch3D Documentation](https://pytorch3d.readthedocs.io/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)


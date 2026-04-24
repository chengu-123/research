import numpy as np
import plotly.graph_objects as go

from io import BytesIO
from PIL import Image

def visualize_voxels(voxels, output_path, return_image=False):
    
    camera_params = {
        "eye": {"x": 1.3960000571786428, "y": -1.5740187908155208, "z": 0.4582309644270989},
        "center": {"x": 0, "y": 0, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1}
    }
    
    # Extract voxel coordinates where the value is 1
    x, y, z = np.where(voxels == 1)
    voxel_colors = 'rgba(255, 122, 0, 1.0)'
    
    # Create a scatter plot for the voxels
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=1,            # Size of the voxel points
                    color=voxel_colors,  # Color of the voxels
                    line=dict(width=0)  # No outline for the voxels,
                ),
            )
        ]
    )
    
    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[0, 63]),  # Set x-axis range
            yaxis=dict(title='Y', range=[0, 63]),  # Set y-axis range
            zaxis=dict(title='Z', range=[0, 63]),  # Set z-axis range
            aspectmode='cube',  # Keep the aspect ratio as a cube
            camera=camera_params
        ),
        title='3D Voxel Visualization',
    )
    
    if return_image:
        buf = BytesIO()
        fig.write_image(buf, width=800, height=800)
        buf.seek(0)
        return np.array(Image.open(buf))
    else: 
        fig.write_image(output_path, width=800, height=800) 
        fig.write_html(output_path.replace('.png', '.html'))

def visualize_voxels_two_parts(voxels_base, voxels_articulated, output_path, return_image=False, grid_normalizer=None):
    
    camera_params = {
        "eye": {"x": 1.3960000571786428, "y": -1.5740187908155208, "z": 0.4582309644270989},
        "center": {"x": 0, "y": 0, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1}
    }
    
    # Extract voxel coordinates where the value is 1
    voxel_colors = ['rgba(255, 122, 0, 1.0)', 'rgba(0, 122, 255, 1.0)']
    scatters = []
    for voxels, colors in zip([voxels_base, voxels_articulated], voxel_colors):
        x, y, z = np.where(voxels == 1)
        scatters.append(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=1, color=colors, line=dict(width=0))))
    
    # Create a scatter plot for the voxels
    fig = go.Figure(data=scatters)
    range_min, range_max = 0, 63
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[range_min, range_max]),  # Set x-axis range
            yaxis=dict(title='Y', range=[range_min, range_max]),  # Set y-axis range
            zaxis=dict(title='Z', range=[range_min, range_max]),  # Set z-axis range
            aspectmode='cube',  # Keep the aspect ratio as a cube
            camera=camera_params
        ),
        title='3D Voxel Visualization',
    )
    
    if return_image:
        buf = BytesIO()
        fig.write_image(buf, width=800, height=800)
        buf.seek(0)
        return np.array(Image.open(buf))
    else:
        fig.write_image(output_path, width=800, height=800)
        fig.write_html(output_path.replace('.png', '.html'))
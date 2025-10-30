import torch
import os
import open3d as o3d
import numpy as np
# from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from typing import NamedTuple
from plyfile import PlyData, PlyElement
import math
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.cm import get_cmap
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from cmocean import cm

"""
TODO: first use the gsplat rasterizer, and if it works, don't need to use the inria rasterizer.
"""


def world_to_cam_means(
    means,
    viewmats
):
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape

    R = viewmats[..., :3, :3]  # [..., C, 3, 3]
    t = viewmats[..., :3, 3]  # [..., C, 3]
    means_c = (
        torch.einsum("...cij,...nj->...cni", R, means) + t[..., None, :]
    )  # [..., C, N, 3]
    return means_c

    
def calculate_global_depth_range(point_clouds, center_position, view_angles, min_vals, max_vals):
    """Calculate global depth range for consistent coloring"""
    
    # Calculate depth reference point (same logic as in your functions)
    if view_angles is not None:
        elev, azim = view_angles
        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)
        camera_distance = np.max(max_vals - min_vals) * 2
        camera_x = camera_distance * np.cos(elev_rad) * np.cos(azim_rad)
        camera_y = camera_distance * np.cos(elev_rad) * np.sin(azim_rad)
        camera_z = camera_distance * np.sin(elev_rad)
        depth_reference_point = np.array([camera_x, camera_y, camera_z])
    else:
        depth_reference_point = np.array([
            (min_vals[0] + max_vals[0]) / 2,
            (min_vals[1] + max_vals[1]) / 2,
            max_vals[2] + (max_vals[2] - min_vals[2])
        ])
    
    # Calculate depths for all frames
    all_depths = []
    
    # Handle both list and array inputs
    if isinstance(point_clouds, list):
        frames = point_clouds
    else:
        frames = [point_clouds[i] for i in range(point_clouds.shape[0])]
    
    for frame in frames:
        # Convert to numpy if needed
        if isinstance(frame, torch.Tensor):
            frame_np = frame.cpu().numpy()
        else:
            frame_np = frame
            
        # Apply centering
        if center_position is not None:
            frame_np = frame_np - np.array(center_position)
            
        # Calculate depths for this frame
        depths = np.sqrt(
            (frame_np[:, 0] - depth_reference_point[0])**2 +
            (frame_np[:, 1] - depth_reference_point[1])**2 +
            (frame_np[:, 2] - depth_reference_point[2])**2
        )
        all_depths.extend(depths)
    
    global_depth_min = np.min(all_depths)
    global_depth_max = np.max(all_depths)
    
    return global_depth_min, global_depth_max, depth_reference_point
    
def pers_proj_means(
    means,  # [..., C, N, 3]
    Ks ,  # [..., C, 3, 3]
    width: int,
    height: int,
):
    """PyTorch implementation of perspective projection for 3D Gaussians."""
    batch_dims = means.shape[:-3]
    C, N = means.shape[-3:-1]
    assert means.shape == batch_dims + (C, N, 3), means.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    
    tx, ty, tz = torch.unbind(means, dim=-1)  # [..., C, N]
    
    # Extract intrinsic parameters
    fx = Ks[..., 0, 0, None]  # [..., C, 1]
    fy = Ks[..., 1, 1, None]  # [..., C, 1]
    cx = Ks[..., 0, 2, None]  # [..., C, 1]
    cy = Ks[..., 1, 2, None]  # [..., C, 1]
    
    # Calculate field of view limits
    tan_fovx = 0.5 * width / fx  # [..., C, 1]
    tan_fovy = 0.5 * height / fy  # [..., C, 1]
    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    
    # Clamp to avoid extreme projections
    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)
    
    # Project to 2D
    means2d = torch.einsum(
        "...ij,...nj->...ni", Ks[..., :2, :3], torch.stack([tx, ty, tz], dim=-1)
    )  # [..., C, N, 2]
    means2d = means2d / tz[..., None]  # [..., C, N, 2]
    
    return means2d

# def setup_camera(w, h, k, w2c, near=0.01, far=100):
#     fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
#     w2c = torch.tensor(w2c).cuda().float()
#     cam_center = torch.inverse(w2c)[:3, 3]
#     w2c = w2c.unsqueeze(0).transpose(1, 2)
#     opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
#                                 [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
#                                 [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
#                                 [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
#     full_proj = w2c.bmm(opengl_proj)
#     cam = Camera(
#         image_height=h,
#         image_width=w,
#         tanfovx=w / (2 * fx),
#         tanfovy=h / (2 * fy),
#         bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
#         scale_modifier=1.0,
#         viewmatrix=w2c,
#         projmatrix=full_proj,
#         sh_degree=0,
#         campos=cam_center,
#         prefiltered=False
#     )
#     return cam

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def setup_camera_blender(w, h, R, T, FoVx, FoVy, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, zfar=100, znear=0.01):
    """
    Return the camera from blender dataset.
    """
    world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)
    camera = Camera(image_height=h,
                    image_width=w,
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
                    scale_modifier=1.0,
                    viewmatrix=world_view_transform.cuda(),
                    projmatrix=full_proj_transform.cuda(),
                    sh_degree=0,
                    campos=camera_center,
                    prefiltered=False)
    return camera
    

def PILtoTorch(pil_image, resolution):
    if resolution is not None:
        resized_image_PIL = pil_image.resize(resolution)
    else:
        resized_image_PIL = pil_image
    if np.array(resized_image_PIL).max()!=1:
        resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    else:
        resized_image = torch.from_numpy(np.array(resized_image_PIL))
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def params2rendervar(params):
    """
    Activate the parameters. Quaternions do not need to be normalized.
    Also need to activate the colors (Nah actually leads to slightly worse performance)
    """
    rendervar = {
        'means': params['means'],
        'colors': params['rgbs'],
        'quats': params['quats'],
        'opacities': torch.sigmoid(params['opacities']),
        'scales': torch.exp(params['scales']),
    }
    return rendervar


def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)

def knn(x, K= 4):
    """
    Return K closest neighbor distances from x.
    """
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)

def params2cpu(params, is_initial_timestep):
    """Detach all params and set to numpy"""
    # if is_initial_timestep:
    #     res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}
    # else:
    #     #THe reason you're saving only means, rgbs, and quats is because scales and opacities are frozen
    #     res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if
    #            k in ['means', 'rgbs', 'quats']}

    #TODO: for now, let's just save everything to make sure there's no mistakes in the code.
    res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}
    return res


def save_params(output_params, exp_path):
    to_save = {}
    for k in output_params[0].keys():
        if k in output_params[1].keys():
            to_save[k] = np.stack([params[k] for params in output_params])
        else:
            to_save[k] = output_params[0][k]
    np.savez(f"{exp_path}/params", **to_save)

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    seg_colors: np.array

def fetchPly(path):
    """
    Load the point clouds from path, use seg_colors is None
    """
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals, seg_colors=None)
    
    
    
def animate_point_clouds(
    point_clouds,
    output_file="point_cloud_animation.mp4",
    center_position=None,
    fps=10,
    point_size=20,
    figsize=(6, 6),
    view_angles=None,
    color='blue',
    is_reverse=True,
    t_subsample=1,
    zoom_factor=0.5,
    flip_x=False,
    flip_y=False,
    flip_z=False,
    use_z_coloring=True,
    use_depth_coloring=False,
    colormap="thermal",
    depth_reference_point=None,
    min_vals=None,
    max_vals=None,
    global_depth_min=None,
    global_depth_max=None
):
    """
    Animate point clouds with optional depth or Z coloring.
    """

    if isinstance(point_clouds, torch.Tensor):
        point_clouds = point_clouds.cpu().numpy()
    
    every_n = int(1/t_subsample)
    point_clouds = point_clouds[::every_n, :, :]
    
    # Get data dimensions
    T, N, _ = point_clouds.shape
    
    # Center point cloud if needed
    if center_position is not None:
        center_position = np.array(center_position)
        point_clouds = point_clouds - center_position

    if is_reverse:
        point_clouds = np.flip(point_clouds, axis=0)
    colormap = getattr(cm, colormap) 
    # Create figure + 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.grid(False)
    
    # Remove 3D panes
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')

    # First frame
    first_frame = point_clouds[0]
    x0, y0, z0 = first_frame[:, 0], first_frame[:, 1], first_frame[:, 2]

    # Handle coloring
    if use_z_coloring:
        scatter = ax.scatter(x0, y0, z0, s=point_size, c=-z0, cmap=colormap, alpha=0.8)

    elif use_depth_coloring:
        if depth_reference_point is None:
            if (min_vals is not None) and (max_vals is not None):
                if view_angles is not None:
                    elev, azim = view_angles
                    elev_rad = np.radians(elev)
                    azim_rad = np.radians(azim)
                    camera_distance = np.max(max_vals - min_vals) * 2
                    camera_x = camera_distance * np.cos(elev_rad) * np.cos(azim_rad)
                    camera_y = camera_distance * np.cos(elev_rad) * np.sin(azim_rad)
                    camera_z = camera_distance * np.sin(elev_rad)
                    depth_reference_point = np.array([camera_x, camera_y, camera_z])
                else:
                    depth_reference_point = np.array([
                        (min_vals[0] + max_vals[0]) / 2,
                        (min_vals[1] + max_vals[1]) / 2,
                        max_vals[2] + (max_vals[2] - min_vals[2])
                    ])
            else:
                raise ValueError("min_vals and max_vals must be provided for depth coloring if no reference point is given")

        depths = np.sqrt(
            (x0 - depth_reference_point[0])**2 +
            (y0 - depth_reference_point[1])**2 +
            (z0 - depth_reference_point[2])**2
        )
        
        # ✅ ADDED: Global depth range support for consistent coloring
        if global_depth_min is not None and global_depth_max is not None:
            scatter = ax.scatter(x0, y0, z0, s=point_size, c=depths, cmap=colormap, 
                               vmin=global_depth_min, vmax=global_depth_max, alpha=0.8)
        else:
            scatter = ax.scatter(x0, y0, z0, s=point_size, c=depths, cmap=colormap, alpha=0.8)

    else:
        scatter = ax.scatter(x0, y0, z0, s=point_size, c=color, alpha=0.8)

    # Axis limits
    ax.set_xlim([-0.12, 0.12])
    ax.set_ylim([-0.12, 0.12])
    ax.set_zlim([-0.07, 0.15])
    
    if flip_x:
        ax.invert_xaxis()
    if flip_y: 
        ax.invert_yaxis()
    if flip_z: 
        ax.invert_zaxis()
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_title('Point Cloud Animation')

    # Initial view
    if view_angles is not None:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])

    def update(frame):
        x, y, z = point_clouds[frame, :, 0], point_clouds[frame, :, 1], point_clouds[frame, :, 2]

        if use_z_coloring:
            scatter._offsets3d = (x, y, z)
            scatter.set_array(-z)

        elif use_depth_coloring:
            depths = np.sqrt(
                (x - depth_reference_point[0])**2 +
                (y - depth_reference_point[1])**2 +
                (z - depth_reference_point[2])**2
            )
            scatter._offsets3d = (x, y, z)
            scatter.set_array(depths)
            
            # ✅ ADDED: Maintain consistent color scaling in animation updates
            if global_depth_min is not None and global_depth_max is not None:
                scatter.set_clim(vmin=global_depth_min, vmax=global_depth_max)

        else:
            scatter._offsets3d = (x, y, z)

        ax.title.set_text(f'Point Cloud Animation - Frame {frame+1}/{T}')
        return scatter,

    # Animate
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps)
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(output_file, writer=writer)
    print(f"Animation saved to {output_file}")
    plt.close()

def animate_point_clouds_lst(point_clouds, output_file="point_cloud_animation.mp4", fps=10, 
                         point_size=20, figsize=(10, 8), view_angles=None, color='blue', is_reverse=False,
                         t_subsample=1, zoom_factor=0.5):
    """
    Animate a sequence of point clouds and save as a video.
    
    Parameters:
    -----------
    point_clouds : list of point clouds
    output_file : str
        Output filename for the video (mp4 format)
    fps : int
        Frames per second for the video
    point_size : int
        Size of points in the visualization
    figsize : tuple
        Figure size (width, height) in inches
    view_angles : list or None
        Initial view angles [elevation, azimuth] if specified
    color : str or array
        Color of the points
    is_reverse: bool
        whether or not the trajectory is reversed
    subsample_rate: float 
        subsampling rate for the tensor, between (0,1)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    from matplotlib.cm import get_cmap
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D

    lst_of_pc_numpy = []
    for pc in point_clouds:
        if isinstance(pc, torch.Tensor):
            lst_of_pc_numpy.append(pc.cpu().numpy())
        else:
            lst_of_pc_numpy.append(pc)

    point_clouds = lst_of_pc_numpy
    # every_n = int(1/t_subsample)
    # point_clouds = point_clouds[::every_n,:,:]
    # Get data dimensions
    # T, N, _ = point_clouds.shape
    first_frame = point_clouds[0]  # Shape: (N, 3)
    min_vals = first_frame.min(axis=0)
    max_vals = first_frame.max(axis=0)
    max_range = max(max_vals - min_vals)
    center = (min_vals + max_vals) / 2
    T = len(point_clouds)
    
    if is_reverse:
        point_clouds =  point_clouds[::-1]
     
    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize scatter plot
    scatter = ax.scatter([], [], [], s=point_size, c=color, alpha=0.8)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
   
    # Fixed limits for Blender scenes
    ax.set_xlim([center[0] - max_range * zoom_factor, center[0] + max_range * zoom_factor])
    ax.set_ylim([center[1] - max_range * zoom_factor, center[1] + max_range * zoom_factor])
    ax.set_zlim([center[2] - max_range * zoom_factor, center[2] + max_range * zoom_factor])

    # Set title
    ax.set_title('Point Cloud Animation')
    
    # Set initial view angle if specified
    if view_angles is not None:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    def update(frame):
        """ Update function for each frame """
        x, y, z = point_clouds[frame][:, 0], point_clouds[frame][:, 1], point_clouds[frame][:, 2]
        
        # Update scatter plot data
        scatter._offsets3d = (x, y, z)
        
        # Update frame title correctly
        ax.title.set_text(f'Point Cloud Animation - Frame {frame+1}/{T}')
        return scatter,
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=T, 
        interval=1000/fps
    )

    # Fixed coordinate system settings
    max_range = 2.0  # Fixed to match [-2,2] scene
    center = np.array([-1, -3, 0])  # make the origin of the axes on the bottom left side
    max_length = max_range * 0.4  # Length of coordinate axes
    
    # X axis - Red
    ax.plot([center[0], center[0] + max_length], [center[1], center[1]], [center[2], center[2]], 'r-', linewidth=2)
    ax.text(center[0] + max_length * 1.1, center[1], center[2], 'X', color='red')

    # Y axis - Green
    ax.plot([center[0], center[0]], [center[1], center[1] + max_length], [center[2], center[2]], 'g-', linewidth=2)
    ax.text(center[0], center[1] + max_length * 1.1, center[2], 'Y', color='green')

    # Z axis - Blue
    ax.plot([center[0], center[0]], [center[1], center[1]], [center[2], center[2] + max_length], 'b-', linewidth=2)
    ax.text(center[0], center[1], center[2] + max_length * 1.1, 'Z', color='blue')

    # Save animation as MP4
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(output_file, writer=writer)
    print(f"Animation saved to {output_file}")
    
    plt.close()
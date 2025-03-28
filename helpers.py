import torch
import os
import open3d as o3d
import numpy as np
# from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from typing import NamedTuple
from plyfile import PlyData, PlyElement
import math
from sklearn.neighbors import NearestNeighbors


"""
TODO: first use the gsplat rasterizer, and if it works, don't need to use the inria rasterizer.
"""



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
    
    
    
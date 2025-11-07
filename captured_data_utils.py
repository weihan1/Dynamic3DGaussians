import os
import json
from tqdm import tqdm
from typing import Any, Dict, List, Optional
from typing_extensions import assert_never

import cv2
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from pycolmap import SceneManager

from torch.utils.data import Sampler
from typing import Union
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    """
    reference: nerf-factory
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene.
    if center_method == "focus":
        # find the closest point to the origin for each camera's center ray
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        # use center of the camera positions
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale

    return transform


def align_principle_axes(point_cloud):
    # Compute centroid
    centroid = np.median(point_cloud, axis=0)

    # Translate point cloud to centroid
    translated_point_cloud = point_cloud - centroid

    # Compute covariance matrix
    covariance_matrix = np.cov(translated_point_cloud, rowvar=False)

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by eigenvalues (descending order) so that the z-axis
    # is the principal axis with the smallest eigenvalue.
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]

    # Check orientation of eigenvectors. If the determinant of the eigenvectors is
    # negative, then we need to flip the sign of one of the eigenvectors.
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    # Create rotation matrix
    rotation_matrix = eigenvectors.T

    # Create SE(3) matrix (4x4 transformation matrix)
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid

    return transform


def transform_points(matrix, points):
    """Transform points using an SE(3) matrix.

    Args:
        matrix: 4x4 SE(3) matrix
        points: Nx3 array of points

    Returns:
        Nx3 array of transformed points
    """
    assert matrix.shape == (4, 4)
    assert len(points.shape) == 2 and points.shape[1] == 3
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(matrix, camtoworlds):
    """Transform cameras using an SE(3) matrix.

    Args:
        matrix: 4x4 SE(3) matrix
        camtoworlds: Nx4x4 array of camera-to-world matrices

    Returns:
        Nx4x4 array of transformed camera-to-world matrices
    """
    assert matrix.shape == (4, 4)
    assert len(camtoworlds.shape) == 3 and camtoworlds.shape[1:] == (4, 4)
    camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds


def normalize(camtoworlds, points=None):
    T1 = similarity_from_cameras(camtoworlds)
    camtoworlds = transform_cameras(T1, camtoworlds)
    if points is not None:
        points = transform_points(T1, points)
        T2 = align_principle_axes(points)
        camtoworlds = transform_cameras(T2, camtoworlds)
        points = transform_points(T2, points)
        return camtoworlds, points, T2 @ T1
    else:
        return camtoworlds, T1

def map_cont_to_int(cont_t, num_images, factor=1):
    """
    Logic: int_t = (cont_t/factor) * (num_images-1)
    """
    int_t = (cont_t /factor)* (num_images - 1)
    int_t = torch.round(int_t).to(torch.int32) 
    return int_t

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, mask_dir: str, resized_image_dir: str, resized_mask_dir: str, factor: int) -> str:
    """Resize image folder and corresponding masks."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_image_dir}.")
    os.makedirs(resized_image_dir, exist_ok=True)
    os.makedirs(resized_mask_dir, exist_ok=True)
    
    image_files = _get_rel_paths(image_dir)
    mask_files = _get_rel_paths(mask_dir)
    
    # dont use this in case hidden files creep up
    # assert len(image_files) == len(mask_files), "should have same number of masks and images"
    
    for image_file, mask_file in tqdm(zip(image_files, mask_files)):
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        #make sure everything ends in png
        resized_image_path = os.path.join(
            resized_image_dir, os.path.splitext(image_file)[0] + ".png"
        )
        #COLMAP ends in jpg.png, so need to splitext twice
        resized_mask_path = os.path.join(
            resized_mask_dir, os.path.splitext(os.path.splitext(mask_file)[0])[0] + ".png"
        )
        
        if os.path.isfile(resized_image_path) and os.path.isfile(resized_mask_path):
            continue
        
        if not os.path.isfile(resized_image_path):
            image = imageio.imread(image_path)[..., :3]  # Take only RGB channels
            resized_size = (
                int(round(image.shape[1] / factor)),  # width
                int(round(image.shape[0] / factor)),  # height
            )
            resized_image = np.array(
                Image.fromarray(image).resize(resized_size, Image.BICUBIC)
            )
            imageio.imwrite(resized_image_path, resized_image)
        
        if not os.path.isfile(resized_mask_path):
            mask = imageio.imread(mask_path)
            resized_mask_size = (
                int(round(mask.shape[1] / factor)),  # width
                int(round(mask.shape[0] / factor)),  # height
            )
            resized_mask = np.array(
                Image.fromarray(mask).resize(resized_mask_size, Image.NEAREST)  # Use NEAREST for masks
            )
            imageio.imwrite(resized_mask_path, resized_mask)
    
    return resized_image_dir, resized_mask_dir



class DynamicParser:
    """COLMAP parser for multiple timesteps.
    NOTE: must remember: timestep 0 must be when the plant is fully grown!!! 
    """

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        align_timesteps: bool = False,
        dates: list= ["08-07-2025"],
        use_dense: bool =False,
        apply_mask:bool=True,
    ):
        
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.align_timesteps = align_timesteps
        self.use_dense = use_dense
        self.apply_mask = apply_mask

        # Initialize storage for all timesteps
        self.timestep_data = [] #this stores per-timestep data
        self.global_transform = np.eye(4)
        
        assert dates is not None, "please specify dates"
        assert all(dates[i] >= dates[i+1] for i in range(len(dates)-1)), "Dates not in decreasing order"
        #Make sure here u go from latest date to earliest date
        self.num_timesteps = len(dates)
        print(f"[DynamicParser] Processing {self.num_timesteps} timesteps...")

        # Process each timestep
        for t, dir_t in enumerate(dates):
            print(f"[DynamicParser] Processing timestep {t}: {dir_t}")
            full_data_dir = os.path.join(self.data_dir, dir_t)
            timestep_parser = self._parse_single_timestep(full_data_dir, t)
            self.timestep_data.append(timestep_parser)

        all_camtoworlds_lst = []
        all_points_lst = []
        for i in range(len(self.timestep_data)):
            all_camtoworlds_lst.append(self.timestep_data[i]["camtoworlds"])
            all_points_lst.append(self.timestep_data[i]["points"])
        all_camtoworlds_array = np.concatenate(all_camtoworlds_lst, axis=0)
        all_points_array = np.concatenate(all_points_lst, axis=0)

        if self.normalize: #normalize cameras to origin
            print("normalizing across all timesteps")
            T1 = similarity_from_cameras(all_camtoworlds_array)
            camtoworlds = transform_cameras(T1, all_camtoworlds_array)
            # visualize_point_cloud(points)
            points = transform_points(T1, all_points_array)
            # visualize_point_cloud(points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            camtoworlds = all_camtoworlds_array  # Use original cameras if not normalizing
            points = all_points_array           # Use original points if not normalizing
            transform = np.eye(4)

        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        scene_scale = np.max(dists)

        # Split the transformed data back to individual timesteps
        start_cam = 0
        start_points = 0
        for i in range(len(self.timestep_data)):
            num_camera_i = all_camtoworlds_lst[i].shape[0]
            num_points_i = all_points_lst[i].shape[0]  # Fixed: was all_points_array[i].shape[0]
            
            # Extract the cameras and points for this timestep
            self.timestep_data[i]["camtoworlds"] = camtoworlds[start_cam:start_cam + num_camera_i]
            self.timestep_data[i]["points"] = points[start_points:start_points + num_points_i]
            
            # Store global parameters
            self.timestep_data[i]["transform"] = transform
            self.timestep_data[i]["scene_scale"] = scene_scale
            
            # Update start indices for next timestep
            start_cam += num_camera_i
            start_points += num_points_i

        # #NOTE: this code is for viz only.
        # import viser
        # try:
        #     print("Starting Viser server...")
        #     server = viser.ViserServer(host="127.0.0.1", port=8009)
        #     print("Viser server created successfully!")
            
        #     server.scene.add_icosphere(
        #         name="hello_sphere",
        #         radius=0.5,
        #         color=(255, 0, 0),
        #         position=(0.0, 0.0, 0.0),
        #     )
        #     print("Scene objects added!")
            
        #     print("Server should be listening now...")
        #     while True:
        #         time.sleep(10.0)
                
        # except Exception as e:
        #     print(f"Error: {e}")
        #     import traceback
        #     traceback.print_exc()
        point_cloud_sequence_untransformed = []
        legend_labels = []
        for t in range(len(self.timestep_data)):
            point_cloud_sequence_untransformed.append(self.timestep_data[t]["points"])
            legend_labels.append(self.timestep_data[t]["data_dir"].split("/")[-1])

        all_cameras = [t["camtoworlds"] for t in self.timestep_data]
        # np.save("all_cameras", np.array(all_cameras, dtype=object))
        # np.save("all_points", np.array(point_cloud_sequence_untransformed, dtype=object))
        # visualization code
        # generate_rotation_sequence(point_cloud_sequence_untransformed, legend_labels=legend_labels,base_filename="pc_rotation", output_dir="rotation_frames")
        # Might need to do this for some point clouds?

        # np.save("all_cameras_after.npy", np.array(all_cameras, dtype=object))
        # np.save("all_points_after.npy", np.array(point_cloud_sequence_untransformed, dtype=object))
        # backwards compatibility, allows us to have the same methods for timestep 0. 
        self._create_unified_data()
        #TODO: export all cameras in a tensor
        print(f"[DynamicParser] Successfully loaded {self.num_timesteps} timesteps")


    def _parse_single_timestep(self, data_dir: str, timestep_idx: int):
        """Parse a single timestep using the original parser logic."""
        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= self.factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // self.factor, cam.height // self.factor)
            mask_dict[camera_id] = None

        print(f"[Timestep {timestep_idx}] {len(imdata)} images, taken by {len(set(camera_ids))} cameras.")

        if len(imdata) == 0:
            raise ValueError(f"No images found in COLMAP for timestep {timestep_idx}.")
        if not (type_ == 0 or type_ == 1):
            print(f"Warning: COLMAP Camera is not PINHOLE for timestep {timestep_idx}. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP
        image_names = [imdata[k].name for k in imdata]
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        #no point in using camera_ids cause we use the same camera
        camera_ids = [camera_ids[i] for i in inds]
        image_ids = inds
        assert sorted(image_names) == image_names, "make sure image_names is sorted"


        # Load extended metadata
        extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                extconf.update(json.load(f))

        # Load bounds
        bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            bounds = np.load(posefile)[:, -2:]

        # Load images
        if self.factor > 1 and not extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{self.factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images_still")
        image_dir = os.path.join(data_dir, "images_still" + image_dir_suffix)
        masks_dir = os.path.join(data_dir, "masks_bg")

        if self.factor > 1:
            resized_mask_dir = masks_dir + image_dir_suffix+"_png"
        else:
            resized_mask_dir = masks_dir 

        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        #remap colmap images names to images names
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if self.factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir, mask_dir = _resize_image_folder(  # downsamples the images with png
                image_dir=colmap_image_dir,
                mask_dir=masks_dir,
                resized_image_dir=image_dir + "_png",
                resized_mask_dir = masks_dir+ image_dir_suffix+"_png",
                factor=self.factor,
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
        masks_paths = sorted([os.path.join(resized_mask_dir, f) for f in os.listdir(resized_mask_dir)])
        assert len(masks_paths) != 0, "we need masks,"
        # 3D points
        if timestep_idx == 0: #where we need the actual point clouds
            if self.use_dense:
                import trimesh
                dense_folder = os.path.join(data_dir, "dense", "fused.ply")
                mesh = trimesh.load(dense_folder) 
                points = mesh.vertices
                points_rgb = mesh.visual.vertex_colors[...,:3] #discard alpha_channel
            else:
                points = manager.points3D.astype(np.float32)
                points_err = manager.point3D_errors.astype(np.float32)
                points_rgb = manager.point3D_colors.astype(np.uint8)
        else: #open the points, but not use them (maybe for viz or smth)
            points = manager.points3D.astype(np.float32)
            points_err = manager.point3D_errors.astype(np.float32)
            points_rgb = manager.point3D_colors.astype(np.uint8)
            

        # points_combined = torch.from_numpy(np.concatenate((points, (points_rgb/255)), axis=-1))
        # save_point_cloud_to_ply(points_combined, "sparse_points.ply")
        point_indices = dict()

        #NOt being used
        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Store normalization transform for this timestep

        # Handle image size correction and undistortion (same as original)
        actual_image = imageio.imread(image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = imsize_dict[camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        
        #adjusting intrinsics based on image shapes
        for camera_id, K in Ks_dict.items(): 
            K[0, :] *= s_width
            K[1, :] *= s_height
            Ks_dict[camera_id] = K
            width, height = imsize_dict[camera_id]
            imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # Undistortion maps
        mapx_dict = dict()
        mapy_dict = dict()
        roi_undist_dict = dict()
        
        for camera_id in params_dict.keys():
            params = params_dict[camera_id]
            if len(params) == 0:
                continue
                
            K = Ks_dict[camera_id]
            width, height = imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]

            mapx_dict[camera_id] = mapx
            mapy_dict[camera_id] = mapy
            Ks_dict[camera_id] = K_undist
            roi_undist_dict[camera_id] = roi_undist
            imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            mask_dict[camera_id] = mask

        # Calculate scene scale

        # Return timestep data as a dictionary
        date = data_dir.split("/")[-1]
        return {
            'date': date,
            'data_dir': data_dir,
            'timestep_idx': timestep_idx,
            'image_names': image_names,
            'image_paths': image_paths,
            'masks_paths': masks_paths,
            'camtoworlds': camtoworlds,
            'camera_ids': camera_ids, #if same camera, this would be all ones
            'image_ids': image_ids, #should be using this to index into poses/images
            'Ks_dict': Ks_dict,
            'params_dict': params_dict, #distortion params
            'imsize_dict': imsize_dict, #used in static_render_traj
            'mask_dict': mask_dict,
            'points': points,
            # 'points_err': points_err,
            'points_rgb': points_rgb, #used for initialization
            # 'point_indices': point_indices,
            # 'transform': transform,
            # 'scene_scale': scene_scale,
            'bounds': bounds,
            'extconf': extconf,
            'mapx_dict': mapx_dict, #distortion params
            'mapy_dict': mapy_dict, #distortion params
            'roi_undist_dict': roi_undist_dict, #distortion params
            'num_images': len(image_names)
        }

    def _create_unified_data(self):
        """Create unified data structures across all timesteps.
        This is fine, since most parameters will only be used for static reconstruction, which 
        is only done for timestep t0.
        """
        # For backward compatibility, expose the first timestep's data at the top level
        first_timestep = self.timestep_data[0]
        for key, value in first_timestep.items():
            if key != 'timestep_idx':
                setattr(self, key, value)
        
        # Add timestep-specific access methods
        self.all_timesteps = self.timestep_data

    def get_timestep_data(self, timestep_idx: int):
        """Get data for a specific timestep."""
        if timestep_idx >= self.num_timesteps:
            raise IndexError(f"Timestep {timestep_idx} out of range [0, {self.num_timesteps-1}]")
        return self.timestep_data[timestep_idx]

    def get_image_at_timestep(self, timestep_idx: int, image_idx: int):
        """Get a specific image at a specific timestep."""
        timestep_data = self.get_timestep_data(timestep_idx)
        return timestep_data['image_paths'][image_idx]

    def get_cameras_at_timestep(self, timestep_idx: int):
        """Get camera poses for a specific timestep."""
        timestep_data = self.get_timestep_data(timestep_idx)
        return timestep_data['camtoworlds']

    def get_points_at_timestep(self, timestep_idx: int):
        """Get 3D points for a specific timestep."""
        timestep_data = self.get_timestep_data(timestep_idx)
        return timestep_data['points']


class Dynamic_Dataset():
    """
    Dynamic dataset for captured data
    """
    def __init__(
        self,
        parser,  # Should be DynamicParser
        split: str = "train",
        is_reverse=False,
        downsample_factor=1, 
        prepend_zero=True,
        include_zero=False, 
        downsample_eval=True,
        cam_batch_size=-1,
        time_normalize_factor=1,
        return_mask=False
    ):
        self.parser = parser
        self.split = split
        self.prepend_zero = prepend_zero
        self.is_reverse = is_reverse
        self.downsample_eval = downsample_eval
        self.cam_batch_size = cam_batch_size
        self.include_zero = include_zero
        self.time_normalize_factor = time_normalize_factor
        self.first_mesh_path = ""
        self.return_mask = return_mask
        self.apply_mask = self.parser.apply_mask
        
        print(f"Normalizing our time from [0, {time_normalize_factor}]")
        
        # Load all images and organize by timestep and camera
        self._load_all_data()
        
        self._setup_split()
    
    def _load_all_data(self):
        """Load all images and camera data across all timesteps"""
        self.timestep_images = {}
        self.timestep_poses = {}
        self.timestep_intrinsics = {}
        self.timestep_masks = {}
        self.timestep_image_paths = {}
        
        camera_id = self.parser.timestep_data[0]["camera_ids"][0]
        
        def load_timestep(t):
            """Load all data for a single timestep - returns the data instead of modifying shared state"""
            timestep_data = self.parser.get_timestep_data(t)
            
            images = {}
            poses = {}
            intrinsics = {}
            masks = {}
            image_paths = []
            
            for i, (image_path, image_id, mask_path) in enumerate(zip(
                timestep_data['image_paths'], 
                timestep_data['image_ids'],
                timestep_data["masks_paths"]
            )):
                # Load mask and invert it
                mask = imageio.imread(mask_path)
                if len(mask.shape) == 3:
                    mask = mask[..., 0]
                # inverted_mask = 255 - mask
                inverted_mask = mask
                inverted_mask = (inverted_mask / 255.0).astype(np.float32)
                
                # Load and process image
                image = imageio.imread(image_path)[..., :3]
                image = (image / 255.0).astype(np.float32)
                
                # Handle undistortion
                params = timestep_data['params_dict'][camera_id]
                if len(params) > 0:
                    mapx, mapy = (
                        timestep_data['mapx_dict'][camera_id],
                        timestep_data['mapy_dict'][camera_id],
                    )
                    image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
                    inverted_mask = cv2.remap(inverted_mask, mapx, mapy, cv2.INTER_NEAREST)
                    
                    x, y, w, h = timestep_data['roi_undist_dict'][camera_id]
                    image = image[y : y + h, x : x + w]
                    inverted_mask = inverted_mask[y : y + h, x : x + w]
                
                if self.apply_mask:
                    image = image * inverted_mask[..., np.newaxis]

                images[image_id] = image
                masks[image_id] = inverted_mask
                poses[image_id] = timestep_data['camtoworlds'][i]
                intrinsics[camera_id] = timestep_data['Ks_dict'][camera_id]
                image_paths.append(os.path.basename(image_path))
            
            del timestep_data
            
            # Return everything as a dict
            return {
                't': t,
                'images': images,
                'masks': masks,
                'poses': poses,
                'intrinsics': intrinsics,
                'image_paths': image_paths
            }
        
        # Load all timesteps in parallel
        max_workers = min(8, os.cpu_count() or 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(load_timestep, t) for t in range(self.parser.num_timesteps)]
            
            # Collect results as they complete
            for future in tqdm(as_completed(futures), total=self.parser.num_timesteps, desc="loading all images"):
                result = future.result()
                t = result['t']
                
                # Now safely update the instance variables (no threading issues here)
                self.timestep_images[t] = result['images']
                self.timestep_masks[t] = result['masks']
                self.timestep_poses[t] = result['poses']
                self.timestep_intrinsics[t] = result['intrinsics']
                self.timestep_image_paths[t] = result['image_paths']
        
        import gc
        gc.collect()
        
        # Validation
        for t in range(self.parser.num_timesteps):
            assert self.timestep_image_paths[t] == sorted(self.timestep_image_paths[t])
        
        print(f"Loaded {self.parser.num_timesteps} timesteps")
        print("")
        print("Printing the first timesteps...")
        for t in range(self.parser.num_timesteps):
            print(f"  Timestep {t}: {len(self.timestep_images[t])} cameras")
    

    def _setup_split(self):
        """
        Setup train/test split based on timesteps
        Pick the first camera of each timestep as test camera, so remove them from self.timestep_poses/images
        Define a "NVS" camera which is not in the union of any cameras of any timesteps
        """
        self.available_timesteps = list(range(self.parser.num_timesteps))
        test_every  = 8
        if self.split == "train":
            # Remove first camera from each timestep for training
            for timestep in self.available_timesteps:
                if len(self.timestep_images[timestep]) > 0:
                    test_image_ids = list(self.timestep_images[timestep].keys())[::test_every]
                    
                    # Remove from training data
                    for image_id in test_image_ids:
                        del self.timestep_images[timestep][image_id]
                        del self.timestep_poses[timestep][image_id]
                        del self.timestep_masks[timestep][image_id]

            print(f"Train split: Using {sum(len(cams) for cams in self.timestep_images.values())} cameras")
            
        #when this runs, it's a new instance of Dynamic_Dataset, so all camera indices still exist.
        elif self.split == "test":
            # Keep only the first camera from each timestep for testing
            for timestep in self.available_timesteps:
                test_image_paths = []
                if len(self.timestep_images[timestep]) > 0:
                    # Get every 10th image_id (already in chronological order)
                    test_image_ids = list(self.timestep_images[timestep].keys())[::test_every]
                    for image_id in test_image_ids:
                        test_image_paths.append(self.timestep_image_paths[timestep][image_id])
                    print(f"the test image names for timestep {timestep} are {test_image_paths}") 
                    # Keep only test images, remove all others
                    all_image_ids = list(self.timestep_images[timestep].keys())
                    train_image_ids = [img_id for img_id in all_image_ids if img_id not in test_image_ids]
                    
                    # Remove training data (keep only test)
                    for image_id in train_image_ids:
                        del self.timestep_images[timestep][image_id]
                        del self.timestep_poses[timestep][image_id]
                        del self.timestep_masks[timestep][image_id]

            print(f"Test split: Using {sum(len(cams) for cams in self.timestep_images.values())} cameras")
            
        else:
            raise ValueError(f"Unknown split: {self.split}. Must be 'train' or 'test'")
        
        self.static_indices = sorted(self.timestep_images[0].keys())
            
    def _compute_nvs_pose(self, margin=0.05):
        world_up = np.array([0,1,0], dtype=np.float32)
        
        # Collect camera centers
        centers = []
        for t in self.available_timesteps:
            for _, Tcw in self.timestep_poses[t].items():
                centers.append(Tcw[:3,3])
        centers = np.array(centers)
        
        center = centers.mean(axis=0)
        ground_y = centers[:,1].min() - margin  # world Y as up
        
        # Horizontal direction: first cam projected onto XZ plane
        hdir = centers[0] - center
        hdir[1] = 0
        hdir /= (np.linalg.norm(hdir) + 1e-9)
        
        radius = np.linalg.norm(hdir)  # simple distance
        pos = center + hdir * radius
        pos[1] = ground_y
        
        # Forward +Z
        forward = center - pos
        forward /= (np.linalg.norm(forward) + 1e-9)
        
        right = np.cross(forward, world_up)
        right /= (np.linalg.norm(right) + 1e-9)
        up_cam = np.cross(right, forward)
        
        T = np.eye(4, dtype=np.float32)
        T[:3,:3] = np.stack([right, up_cam, forward], axis=1)
        T[:3,3] = pos
        return T


    def num_timesteps(self) -> int:
        """Return the number of timesteps"""
        return self.parser.num_timesteps
    
    def __len__(self):
        """Return the number of available timesteps"""
        return self.num_timesteps()
    
    def __getitem__(self, timestep: int) -> Dict[str, Any]:
        """
        Retrieve data for a single timestep. 
        NOTE: For single timestep, we output a dictionary, incompatible with raster_params
        NOTE: here we don't take into account the cam batch size and just output all cameras.
        """
        if timestep not in self.timestep_images:
            raise IndexError(f"Timestep {timestep} not found. Available: {list(self.timestep_images.keys())}")
        
        data = {
            "K": {},
            "camtoworld": {},
            "image": {},
            "image_id": {}
        }
        
        #For the intrinsics, just set the key to be 1
        data["K"][1] = torch.from_numpy(self.timestep_intrinsics[timestep][1]).float() 

        # Get all cameras available at this timestep
        for i, camera_id in enumerate(self.timestep_images[timestep]):
            data["camtoworld"][camera_id] = torch.from_numpy(self.timestep_poses[timestep][camera_id]).float()
            data["image"][camera_id] = torch.from_numpy(self.timestep_images[timestep][camera_id]).float()
            data["K"][camera_id] = torch.from_numpy(self.timestep_intrinsics[timestep][1]) #intrinsics is always just indexed at 1
            data["image_id"][camera_id] = i #(always just use numbers 0 -> len(images)- 1 as image ids)
         
        return data
    
    def __getitems__(self, timesteps: Union[List[int], np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 
        Retrieve data for multiple timesteps as a batch.
        Just returns all cameras available for the queried timesteps.
        The only useful stuff for us is 
        1) self.timestep_images (timestep -> images)
        2) self.timestep_poses (timestep -> poses)

        Returns:
            Tuple:
            - `c2ws`: (N, 4, 4) tensor of camera-to-world matrices for all views.
            - `gt_images`: (N, T, H, W, C) tensor of images for all views at selected timesteps.
            - `inp_t`: (T,) tensor of normalized time values
        NOTE: This assumes all timesteps have the same cameras available
        """
        if isinstance(timesteps, np.ndarray):
            timesteps = timesteps.tolist()
        
        
        # Handle include_zero option
        timesteps_to_use = timesteps.copy()
        if self.include_zero and 0 not in timesteps_to_use:
            timesteps_to_use.insert(0, 0)
        
        #TODO: right now, just to make it simple, we make it only work on temp_batch_size == 1
        #TODO: otherwise, we might need to return a list of tensors, which might be expensive
        
        selected_timestep = timesteps[0] 
        available_cameras = (self.timestep_poses[selected_timestep])
        # Apply camera batching
        if self.cam_batch_size == -1:  # Use all available cameras
            available_cameras = (self.timestep_poses[selected_timestep])
            selected_cameras = available_cameras
        else: #sample number of cameras from the set of available cameras at that timestep
            available_cameras = sorted(self.timestep_poses[selected_timestep])
            selected_cameras = random.sample(available_cameras, 
                                           min(self.cam_batch_size, len(available_cameras)))
        
        c2w_list = []
        img_list = []
        masks_list = []
        
        
        # Collect data for each selected camera
        for camera_id in selected_cameras:
            c2w_list.append(torch.from_numpy(self.timestep_poses[selected_timestep][camera_id]).float())
            img_list.append(torch.from_numpy(self.timestep_images[selected_timestep][camera_id]).float())
            masks_list.append(torch.from_numpy(self.timestep_masks[selected_timestep][camera_id]).float())
            
        
        c2ws = torch.stack(c2w_list)  # (N, 4, 4)
        gt_images = torch.stack(img_list)[:, None]  # (N, T, H, W, C)
        gt_masks = torch.stack(masks_list)[:, None]
        
        #Normalize the timesteps in [0,...,self.time_normalize_factor]
        cont_t = torch.tensor(timesteps, dtype=torch.float32) / (self.num_timesteps() - 1)
        cont_t *= self.time_normalize_factor
        
        # Handle prepend_zero option
        if not self.include_zero:  # if we included zero earlier we don't include it now
            if self.prepend_zero:
                inp_t = torch.cat((torch.tensor([0.0]), cont_t), dim=0)
            else:
                inp_t = cont_t
        else:
            inp_t = cont_t

        if self.return_mask:
            return c2ws, gt_images, inp_t, gt_masks
        else:
            return c2ws, gt_images, inp_t
    
    def get_available_cameras_at_timestep(self, timestep: int) -> List:
        """Return list of available camera IDs at a specific timestep"""
        if timestep in self.timestep_images:
            return list(self.timestep_images[timestep].keys())
        return []
    
    def get_timestep_range(self) -> tuple[int, int]:
        """Return the range of available timesteps"""
        return (0, self.num_timesteps() - 1)
    
    def getfirstcam(self, timesteps):
        """
        Get 
        """

    def custom_collate_fn(self, batch):
        """
        Custom collate function for the DynamicDataset's __getitems__ method.
        Args:
        -batch: tuple of c2ws, gt_images, inp_t where c2ws is of shape (N, 4, 4) and gt_images is of shape
        (N, T_batch, 400, 400, 4), and inp_t is of shape (T_batch+1,)
        NOTE: if you sort the inp_t, then need to sort the gt images.
        """
        c2ws_batch = batch[0]
        gt_images_batch = batch[1]
        inp_t = batch[2]
        gt_masks_batch = batch[3]

        #Sort inp_t without the first element (which is 0)
        t0, inp_t_to_sort = inp_t[0], inp_t[1:]
        sorted_inp_t, indices = inp_t_to_sort.sort()

        #Sort the image batch and the masks
        gt_images_batch = gt_images_batch[:, indices, ...]
        gt_masks_batch = gt_masks_batch[:, indices, ...]

        #Recreate the new inp_t
        new_inp_t = torch.cat((t0.unsqueeze(0), sorted_inp_t), dim=0)
        int_t = map_cont_to_int(new_inp_t,self.num_timesteps())

        return c2ws_batch, gt_images_batch, new_inp_t, int_t, gt_masks_batch


class SingleTimeDataset(Dataset):
    def __init__(self, dataset, timestep):
        """
        Custom single time dataset for specific timestep training.
        len() would return the number of cameras (images) for that timestep.
        Indexing into this dataset will return one of the cameras for that timestep.
        """
        self.dataset = dataset
        self.timestep = timestep

    def __len__(self):
        return len(self.dataset[self.timestep]["image_id"])


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve data for a single timestep. 
        Here idx stands for a single image.
        NOTE: For single timestep, we output a dictionary, incompatible with raster_params
        NOTE: for statictimedataset, self.timestep = 0
        """
        idx = self.dataset.static_indices[idx]
        data = dict(
            K = torch.from_numpy(self.dataset.timestep_intrinsics[self.timestep][1]).float(),
            camtoworld = torch.from_numpy(self.dataset.timestep_poses[self.timestep][idx]).float(),
            image = torch.from_numpy(self.dataset.timestep_images[self.timestep][idx]).float(),
            image_id = idx,
            mask = torch.from_numpy(self.dataset.timestep_masks[self.timestep][idx]).float()
        ) 
        
        return data


def initialize_params_captured(parser, scene_scale, learn_masks):
    """
    Use this when training on custom plant dataset.
    It has "camera_angle_x" and "frames" as keys
    1. Load images
    2. Initialize gaussian parameters
    """
    init_scale = 1.0
    init_opacity = 0.1
    from helpers import knn 
    #Initialize using gt geometry
    points = torch.from_numpy(parser.points).float()
    rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    print(f"points are initialized in [{points.min(), points.max()}]")
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = {
        'means': points,
        'rgbs': rgbs,
        'quats':  quats, 
        'opacities': opacities,
        'scales': scales,
    }
    if learn_masks:
        masks = torch.zeros((points.shape[0], 1), dtype=torch.float32, device="cuda")
        params["masks"] = masks

    params = {k: torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}

    
    variables = {'scene_radius': scene_scale,
                 'fixed_bkgd': torch.zeros(1, 3, device="cuda")} #fixed black background we use for rendering eval for instance.

    return params, variables


class InfiniteNeuralODEDataSampler(Sampler):
    """
    Infinite sampler that endlessly yields indices from the provided set
    If shuffle is set to False, then if my list is [1,2,3,4,5,6,7] with batch size of 2,
    it will always sample [1,2], [3,4], [5,6], [7,1] (wrapping around)
    """
    def __init__(self, data_source, my_indices, shuffle=False):
        self.data_source = data_source
        if isinstance(my_indices, torch.Tensor):
            self.indices = my_indices.tolist()
        else:
            self.indices = my_indices
        self.shuffle = shuffle
        
    def __iter__(self):
        while True:  # Create an infinite iterator
            if self.shuffle:
                # Create a copy of indices to avoid modifying the original
                indices = self.indices.copy() if hasattr(self.indices, 'copy') else self.indices[:]
                random.shuffle(indices)
                yield from indices
            else:
                yield from self.indices
    
    def __len__(self):
        # This isn't strictly accurate for an infinite sampler,
        # but is needed for some PyTorch internals
        return 1000000  # A large number



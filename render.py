import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5;8.6'
import torch
import json
import copy
import numpy as np
from PIL import Image
import random
from random import randint
from tqdm import tqdm
from torch import Tensor
from typing import Optional, Tuple, Dict
from collections import defaultdict
from datetime import datetime
from glob import glob
from gsplat import rasterization

# Use our own rasterizer for fair comparison
# from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params, world_to_cam_means, pers_proj_means

from external import calc_psnr, build_rotation, densify, update_params_and_optimizer
from argparse import ArgumentParser
import imageio.v2 as imageio
from helpers import knn, animate_point_clouds, animate_point_clouds_lst, calculate_global_depth_range

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
import seaborn as sns
from cmocean import cm
from interpolator import GaussianInterpolator

def visualize_point_cloud(
    point_clouds,
    output_file="point_cloud.png",
    center_position=None,
    subsample_factor=1,
    point_size=20,
    figsize=(6, 6),
    min_vals=None,
    max_vals=None,
    view_angles=None,
    depth_reference_point=None,
    color="darkgray",
    use_z_coloring=True,  
    use_depth_coloring=False,
    colormap='thermal',   
    zoom_factor=0.5,
    flip_x=False,
    flip_y=False,
    flip_z=False,
    global_depth_min=None,
    global_depth_max=None,
    show_convex_hull=False,
    hull_alpha=0.1,
    hull_color='blue',
    hull_edge_color='darkblue',
    hull_edge_width=0.5
):
    
    if isinstance(point_clouds, torch.Tensor):
        point_clouds = point_clouds.cpu().numpy()
    point_clouds = point_clouds[::subsample_factor,:]
    N = point_clouds.shape[0]
    
    if center_position is not None:
        center_position = np.array(center_position)
        point_clouds = point_clouds - center_position
    
    if min_vals is None or max_vals is None:
        exit("need to set them to ensure consistency")

    colormap = getattr(cm, colormap)
    # Create figure with minimal margins
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # REMOVE ALL WHITESPACE AND VISUAL ELEMENTS
    ax.set_axis_off()
    ax.grid(False)
    
    # Make 3D panes invisible
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Extract coordinates
    x = point_clouds[:, 0]
    y = point_clouds[:, 1]
    z = point_clouds[:, 2]
    
    # Add convex hull if requested
    if show_convex_hull and len(point_clouds) >= 4:  # Need at least 4 points for 3D hull
        try:
            # Compute 3D convex hull
            hull = ConvexHull(point_clouds)
            
            # Create triangular faces for the hull
            hull_faces = []
            for simplex in hull.simplices:
                # Each simplex is a triangle (3 vertices)
                triangle = point_clouds[simplex]
                hull_faces.append(triangle)
            
            # Add hull faces to the plot
            hull_collection = Poly3DCollection(hull_faces, 
                                             alpha=hull_alpha, 
                                             facecolor=hull_color,
                                             edgecolor=hull_edge_color,
                                             linewidth=hull_edge_width)
            ax.add_collection3d(hull_collection)
            
        except Exception as e:
            print(f"Warning: Could not compute convex hull - {e}")
    
    # Create scatter plot with appropriate coloring
    if use_z_coloring:
        scatter = ax.scatter(x, y, z, s=point_size, c=-z, cmap=colormap, alpha=0.8)
    elif use_depth_coloring:
        # Calculate depth from camera/viewpoint
        if depth_reference_point is None:
            # Use camera position based on view angles
            if view_angles is not None:
                elev, azim = view_angles
                # Convert spherical to cartesian for camera position
                elev_rad = np.radians(elev)
                azim_rad = np.radians(azim)
                
                # Calculate camera distance (you may want to adjust this)
                camera_distance = np.max(max_vals - min_vals) * 2
                
                camera_x = camera_distance * np.cos(elev_rad) * np.cos(azim_rad)
                camera_y = camera_distance * np.cos(elev_rad) * np.sin(azim_rad)
                camera_z = camera_distance * np.sin(elev_rad)
                
                depth_reference_point = np.array([camera_x, camera_y, camera_z])
            else:
                # Default to center of bounding box + offset in Z
                depth_reference_point = np.array([
                    (min_vals[0] + max_vals[0]) / 2,
                    (min_vals[1] + max_vals[1]) / 2,
                    max_vals[2] + (max_vals[2] - min_vals[2])
                ])
        
        # Calculate distances from reference point
        depths = np.sqrt(
            (x - depth_reference_point[0])**2 + 
            (y - depth_reference_point[1])**2 + 
            (z - depth_reference_point[2])**2
        )
        
        # ✅ FIXED: Proper if/else structure
        if global_depth_min is not None and global_depth_max is not None:
            scatter = ax.scatter(x, y, z, s=point_size, c=depths, cmap=colormap, 
                            vmin=global_depth_min, vmax=global_depth_max, alpha=0.8)
        else:
            scatter = ax.scatter(x, y, z, s=point_size, c=depths, cmap=colormap, alpha=0.8)
    else:
        scatter = ax.scatter(x, y, z, s=point_size, c=color, alpha=0.8)
    
    # Set view angle
    if view_angles is not None:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Set limits
    ax.set_xlim([-0.12, 0.12])
    ax.set_ylim([-0.12, 0.12])
    ax.set_zlim([-0.07, 0.15])
    
    if flip_x:
        ax.invert_xaxis()
    if flip_y:
        ax.invert_yaxis()
    if flip_z:
        ax.invert_zaxis()
    
    # Remove all subplot margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save with minimal padding
    plt.savefig(output_file, dpi=300, pad_inches=0,  # Removed bbox_inches="tight"
                facecolor='none', edgecolor='none')
    print(f"Visualization saved to {output_file}")
    
    plt.close()
    return fig, ax



def find_closest_gauss(gt, gauss, return_unique=False, batch_size=1024):
    gt = torch.tensor(gt, dtype=torch.float32)
    gauss = torch.tensor(gauss, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt = gt.to(device)
    gauss = gauss.to(device)

    indices = []

    for i in range(0, len(gt), batch_size):
        gt_batch = gt[i:i+batch_size]  # (B, 3)
        dists = torch.cdist(gt_batch, gauss)  # (B, N_gauss)
        min_idx = torch.argmin(dists, dim=1)  # (B,)
        indices.append(min_idx)

    indices = torch.cat(indices, dim=0).cpu()

    if return_unique:
        return torch.unique(indices)
    else:
        return indices

def get_dataset(t, images, cam_to_worlds_dict, intrinsics):
    """
    Retrieve all cameras for timestep t.
    If t != 0, only retrieve the cameras from selected_cam_ids. We want to simulate 27 cameras on last timestep and only 9 
    otherwise.
    """
    data = {
        "K": intrinsics,
        "camtoworld": {},
        "image": {},
        "image_id": {}
    }
    all_camera_indices = list(images.keys())
    height, width = images[all_camera_indices[0]][0].shape[0], images[all_camera_indices[0]][0].shape[1]
    
    for camera_id, img_list in images.items():
        data["camtoworld"][camera_id] = cam_to_worlds_dict[camera_id]
        data["image"][camera_id] = torch.from_numpy(img_list[t]).float() #retrieves 
        # data["image_id"][camera_id] = self.image_ids_dict[camera_id][timestep]
    data["width"] = width
    data["height"] = height 
    return data




def initialize_params_and_get_data(data_dir, md, every_t, render_white=False,num_gaussians=100_000):
    """
    Use this when training on custom plant dataset.
    It has "camera_angle_x" and "frames" as keys
    1. Load images
    """
    cam_to_worlds_dict = {} #we just need on cam to worlds per view
    images = {} #(N: T, H, W, C)
    alpha_masks = {}
    image_ids_dict = {} 
    images_ids_lst = [] #only used for counting
    progress_bar = tqdm(total=len(md["frames"]), desc=f"Loading the data from {data_dir}")
    for frame in md["frames"]: 
        image_ids = frame["file_path"].replace("./", "")
        file_path = os.path.join(data_dir, image_ids)
        camera_id = image_ids.split("/")[-2]
        img = imageio.imread(file_path)
        norm_img = img/255.0
        norm_img[norm_img<0.1] = 0 #get rid of some background noise from blender scenes
        norm_img = np.clip(norm_img, 0, 1) 
        alpha_mask = norm_img[..., 3:4] #(h,w,1)
        if camera_id not in images:
            images[camera_id] = []
            images[camera_id].append(norm_img)
            image_ids_dict[camera_id] = []
            image_ids_dict[camera_id].append(image_ids)
            alpha_masks[camera_id] = []
            alpha_masks[camera_id].append(alpha_mask)
        else:
            images[camera_id].append(norm_img)
            image_ids_dict[camera_id].append(image_ids)
            alpha_masks[camera_id].append(alpha_mask)

        c2w = torch.tensor(frame["transform_matrix"])
        c2w[0:3, 1:3] *= -1  # Convert from OpenGL to OpenCV

        cam_to_worlds_dict[camera_id] = c2w
        progress_bar.update(1)
        images_ids_lst.append(image_ids)

    progress_bar.close()

    # Retrieve camera intrinsics
    image_height, image_width = images[list(images.keys())[0]][0].shape[:2] #select the first image from first view cx = image_width / 2.0
    cx = image_width / 2.0
    cy = image_height / 2.0
    fl = 0.5 * image_width / np.tan(0.5 * md["camera_angle_x"])
    intrinsics = torch.tensor(
        [[fl, 0, cx], [0, fl, cy], [0, 0, 1]], dtype=torch.float32
    )

    #All cameras
    unique_cameras_lst = [v for _, v in cam_to_worlds_dict.items()]
    camera_locations = np.stack(unique_cameras_lst, axis=0)[:, :3, 3]
    scene_center = np.mean(camera_locations, axis=0)
    dists = np.linalg.norm(camera_locations - scene_center, axis=1)
    trainset_scene_scale = np.max(dists)
    global_scale = 1.0
    scene_scale = trainset_scene_scale * 1.1 * global_scale
    
    variables = {'scene_scale': scene_scale,
                 'fixed_bkgd': torch.zeros(1, 3, device="cuda")} #fixed black background we use for rendering eval for instance.

    if render_white:
        return variables, images, cam_to_worlds_dict, intrinsics, alpha_masks
    else:
        return variables, images, cam_to_worlds_dict, intrinsics


def rasterize_splats(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    masks: Optional[Tensor] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor, Dict]:
    """
    Rasterize the splats, assume they are already activated
    module with the color of the splat.

    Return 
    -Rasterized image
    -Rendered opacity mask 
    -Additional rendering infos
    """
    assert opacities.max() <= 1 and opacities.min() >= 0, "make sure parameters are activated a priori"
    #activate scales and opacities
    rasterize_mode = "classic"
    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
        Ks=Ks,  # [C, 3, 3]
        width=width,
        height=height,
        packed=False,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode=rasterize_mode,
        **kwargs,
    )
    if masks is not None:
        render_colors[~masks] = 0
    return render_colors, render_alphas, info


@torch.no_grad()
def render_imgs(dataset, params, variables, timestep):
    """
    Renders all images for the cameras in the dataset
    """
    curr_data = {"Ks": dataset["K"].unsqueeze(0).to("cuda"),
                 "width": dataset["width"],
                 "height": dataset["height"]}

    all_camera_indices = list(dataset["camtoworld"])
    # elapsed_time = 0
    num_cameras = len(all_camera_indices)
    first_available_camera_idx = all_camera_indices[0]
    h,w = dataset["image"][first_available_camera_idx].shape[0], dataset["image"][first_available_camera_idx].shape[1]

    pred_images = torch.zeros(num_cameras, h, w, 3)
    for i, camera_indx in enumerate(all_camera_indices):
        selected_c2w = dataset["camtoworld"][camera_indx]
        selected_image = dataset["image"][camera_indx]
        curr_data["c2w"] = selected_c2w.to("cuda")
        curr_data["image"] = selected_image.to("cuda") 
        rendervar = params2rendervar(params)
        means = rendervar["means"]
        colors = rendervar["colors"]
        quats = rendervar["quats"]
        opacities = rendervar["opacities"]
        scales = rendervar["scales"]
        im, alphas, info = rasterize_splats(means, quats, scales, opacities, colors,
                                                camtoworlds=curr_data["c2w"][None], Ks=curr_data["Ks"],
                                                width=curr_data["width"], height=curr_data["height"])
        gt_image = curr_data["image"]
        fixed_bkgd = variables["fixed_bkgd"]
        gt_has_alpha = gt_image.shape[-1] == 4
        if gt_has_alpha:
            bkgd = fixed_bkgd
            gt_alpha = gt_image[..., [-1]]
            gt_rgb = gt_image[..., :3]
            eval_pixels = gt_rgb * gt_alpha + bkgd * (1 - gt_alpha) #(1, H, W, 3)
            eval_colors = im + bkgd * (1.0 - alphas) #(1, H, W, 3)
            image_pixels = gt_image 
            image_colors = torch.cat([im, alphas], dim=-1)
            eval_colors = torch.clamp(eval_colors, 0.0, 1.0)
            eval_pixels = torch.clamp(eval_pixels, 0.0, 1.0)
            image_colors = torch.clamp(image_colors, 0.0, 1.0).squeeze()

        else:
            image_pixels = gt_image 
            eval_pixels = gt_image 
            im = torch.clamp(im, 0.0, 1.0)
            image_colors =im 
            eval_colors =im 

        #Compute masked psnr 
        pred_image = eval_colors.squeeze() #(H,W,3)
        pred_images[i] = pred_image
    
    return pred_images 


@torch.no_grad() 
def render_eval(exp_path, data_dir, every_t, split):
    """Render eval stuff"""
    md_path =  f"{data_dir}/transforms_{split}.json"
    with open(md_path, "r") as f:
        md = json.load(f)

    variables, images, cam_to_worlds_dict, intrinsics = initialize_params_and_get_data(data_dir, md, every_t)
    param_path = glob(f"{exp_path}/*.npz")[0]
    param_dict = np.load(param_path) #each param is of shape (T, N, F)
    params = {}

    output_folder = os.path.join(exp_path, split)
    os.makedirs(output_folder, exist_ok=True)

    for k,v in param_dict.items():
        params[k] = torch.tensor(v).to("cuda")
    num_timesteps = params["means"].shape[0]
    render_img_frames = np.zeros((num_timesteps, len(images.keys()), 400, 400, 3))
    
    cam_indices_lst = list(images.keys())
    for cam_idx in cam_indices_lst:
        full_cam_path = os.path.join(output_folder, cam_idx)
        os.makedirs(full_cam_path, exist_ok=True)

    gt_tracks_path = None
    scene = None
    gt_idxs_viz_pc = None
    render_tracks = True
    if split == "test":
        # Determine scene name
        scene_names = ["plant_1", "plant_2", "plant_3", "plant_4", "plant_5", 
                      "rose", "lily", "tulip", "clematis", "peony"]
        for scene_name in scene_names:
            if scene_name in data_dir:
                scene = scene_name
                break
        
        gt_tracks_path = os.path.join(data_dir, "meshes", f"relevant_{scene}_meshes", "trajectory_frames.npz")
        if os.path.exists(gt_tracks_path):
            gt_tracks = np.load(gt_tracks_path)
            gt_t0_all = gt_tracks["frame_0000"]  # (N,3)
            
            # Setup for point cloud visualization

    #Create all folders/subfolders
    if render_tracks and gt_tracks_path and os.path.exists(gt_tracks_path):
        tracks_output_folder = os.path.join(output_folder, "tracks")
        os.makedirs(tracks_output_folder, exist_ok=True)
        
        # Track visualization parameters
        if "plant" not in scene:
            subsample_factor_tracks = 15 #prevent overcrowding (10k points is good)
        else:
            subsample_factor_tracks = 1 #the plants scene have fewer mesh vertices so no need to subsample
        gt_t0 = gt_t0_all[::subsample_factor_tracks] #make flows less crowded
        opacity_threshold_flow = 0.3
        tracking_window = 5
        arrow_thickness = 2
        flow_skip = 1
        show_only_visible = True
        fade_strength = 0.8
        min_alpha = 0.1
        
        print(f"GT tracks has {gt_t0.shape[0]} points")
        
        #Using the same gt_idxs
        camera_track_data = {}
        gt_idxs = find_closest_gauss(gt_t0, params["means"][0, :, :3].cpu().numpy())
        if show_only_visible:
            opacities_idxs = torch.sigmoid(params["opacities"][0, gt_idxs]).cpu()
            gt_idxs = gt_idxs[opacities_idxs > opacity_threshold_flow]
        n_gaussians = gt_idxs.shape[0]
        cmap = plt.cm.get_cmap("jet") #following tracking everything
        colors = []
        for i in range(n_gaussians):
            color = cmap(i / n_gaussians)[:3]  # Get RGB, ignore alpha
            colors.append((int(color[0]*255), int(color[1]*255), int(color[2]*255)))

        # Initialize tracking variables per camera
        for cam_idx in cam_indices_lst:
            camera_track_data[cam_idx] = {
                'gt_idxs': gt_idxs,
                'all_trajs': None,
                'all_times': None,
                'colors': colors,
                'track_imgs': [],
                'track_path': os.path.join(tracks_output_folder, cam_idx)
            }
            os.makedirs(camera_track_data[cam_idx]['track_path'], exist_ok=True)

    #main rendering loop
    for t in tqdm(range(num_timesteps)):
        dataset = get_dataset(t, images, cam_to_worlds_dict, intrinsics) #getting all cameras for time t
        time_params = {k: v[t] for k,v in params.items()}
        pred_images = render_imgs(dataset, time_params, variables, t)
        render_img_frames[t] = pred_images.cpu().numpy()
        for i, img in enumerate(pred_images): #loop over each camera
            full_cam_path = os.path.join(output_folder, cam_indices_lst[i])
            full_image_path = os.path.join(full_cam_path, f"{t:05d}.png")
            imageio.imwrite(full_image_path, (img.cpu().numpy()*255).astype(np.uint8))

        if render_tracks and gt_tracks_path and os.path.exists(gt_tracks_path):
            current_means3d = params["means"][t, :, :3]  # (N, 3)
            
            for i, cam_idx in enumerate(cam_indices_lst):
                cam_data = camera_track_data[cam_idx]
                
                # Get current rendering
                current_rendering = (pred_images[i].cpu().numpy() * 255).astype(np.uint8).copy()
                
                # Initialize gt_idxs for this camera if not done
                
                # Get current camera viewmat
                current_c2w = dataset["camtoworld"][cam_idx]
                current_viewmat = torch.linalg.inv(current_c2w.to("cuda"))
                
                # Project current 3D positions to 2D
                selected_means = current_means3d[cam_data['gt_idxs']]
                current_means_cam = world_to_cam_means(selected_means, current_viewmat[None])
                means_2d = pers_proj_means(current_means_cam, dataset["K"][None].to("cuda"), 400, 400)
                current_projections = means_2d.squeeze()
                
                # Update 3D trajectories
                if cam_data['all_trajs'] is None:
                    cam_data['all_times'] = np.array([t])
                    cam_data['all_trajs'] = selected_means.unsqueeze(0).cpu().numpy()
                else:
                    cam_data['all_times'] = np.concatenate((cam_data['all_times'], np.array([t])), axis=0)
                    cam_data['all_trajs'] = np.concatenate((cam_data['all_trajs'], selected_means.unsqueeze(0).cpu().numpy()), axis=0)
                
                # Create trajectory visualization
                current_projections_np = current_projections.cpu().numpy()
                n_gaussians = len(cam_data['gt_idxs'])
                
                # Mask for valid projections
                current_mask = (current_projections_np[:, 0] >= 0) & (current_projections_np[:, 0] < 400) & \
                              (current_projections_np[:, 1] >= 0) & (current_projections_np[:, 1] < 400)
                
                # Draw current points
                for idx in range(0, n_gaussians, flow_skip):
                    if current_mask[idx]:
                        color_idx = (idx // flow_skip) % len(cam_data['colors'])
                        cv2.circle(current_rendering, 
                                 (int(current_projections_np[idx, 0]), int(current_projections_np[idx, 1])), 
                                 2, cam_data['colors'][color_idx], -1)
                
                # Draw trajectories if we have multiple frames
                if cam_data['all_trajs'].shape[0] > 1:
                    traj_img = np.ascontiguousarray(np.zeros((400, 400, 3), dtype=np.uint8))
                    
                    # Apply tracking window
                    all_trajs = cam_data['all_trajs']
                    all_times = cam_data['all_times']
                    if tracking_window is not None and tracking_window < all_trajs.shape[0]:
                        all_trajs = all_trajs[-tracking_window:]
                        all_times = all_times[-tracking_window:]
                    
                    # Draw trajectory lines
                    for t_idx in range(all_trajs.shape[0] - 1):
                        prev_gaussians = torch.from_numpy(all_trajs[t_idx]).to("cuda")
                        prev_projections_cam = world_to_cam_means(prev_gaussians, current_viewmat[None])
                        prev_projections = pers_proj_means(prev_projections_cam, dataset["K"][None].to("cuda"), 400, 400)
                        prev_projections = prev_projections.squeeze()
                        prev_time = all_times[t_idx]
                        
                        curr_gaussians = torch.from_numpy(all_trajs[t_idx + 1]).to("cuda")
                        curr_projections_cam = world_to_cam_means(curr_gaussians, current_viewmat[None])
                        curr_projections = pers_proj_means(curr_projections_cam, dataset["K"][None].to("cuda"), 400, 400)
                        curr_projections = curr_projections.squeeze()
                        curr_time = all_times[t_idx + 1]
                        
                        # Calculate fade factor
                        time_diff = t - curr_time
                        max_time_diff = t - all_times[0] if len(all_times) > 1 else 1
                        
                        if max_time_diff > 0:
                            fade_factor = 1.0 - (time_diff / max_time_diff) * fade_strength
                            fade_factor = max(fade_factor, min_alpha)
                        else:
                            fade_factor = 1.0
                        
                        # Get masks for valid 2D projections
                        prev_projections_np = prev_projections.cpu().numpy()
                        curr_projections_np = curr_projections.cpu().numpy()
                        
                        prev_mask = (prev_projections_np[:, 0] >= 0) & (prev_projections_np[:, 0] < 400) & \
                                   (prev_projections_np[:, 1] >= 0) & (prev_projections_np[:, 1] < 400)
                        curr_mask = (curr_projections_np[:, 0] >= 0) & (curr_projections_np[:, 0] < 400) & \
                                   (curr_projections_np[:, 1] >= 0) & (curr_projections_np[:, 1] < 400)
                        
                        # Draw trajectory lines
                        if curr_time <= t and prev_time <= t:
                            for idx in range(0, curr_projections.shape[0], flow_skip):
                                color_idx = (idx // flow_skip) % len(cam_data['colors'])
                                if prev_mask[idx] and curr_mask[idx]:
                                    faded_color = tuple(int(c * fade_factor) for c in cam_data['colors'][color_idx])
                                    traj_img = cv2.line(traj_img,
                                                       (int(prev_projections_np[idx, 0]), int(prev_projections_np[idx, 1])),
                                                       (int(curr_projections_np[idx, 0]), int(curr_projections_np[idx, 1])),
                                                       faded_color, arrow_thickness)
                    
                    # Overlay trajectories on rendering
                    current_rendering[traj_img > 0] = traj_img[traj_img > 0]
                
                # Save track visualization
                track_image_path = os.path.join(cam_data['track_path'], f"{t:05d}.png")
                imageio.imwrite(track_image_path, current_rendering)
                cam_data['track_imgs'].append(current_rendering)

    #saving tracks
    fps = len(render_img_frames)/3
    if render_tracks and gt_tracks_path and os.path.exists(gt_tracks_path):
        for cam_idx in cam_indices_lst:
            cam_data = camera_track_data[cam_idx]
            if cam_data['track_imgs']:
                track_video_path = f"{cam_data['track_path']}/full_track.mp4"
                imageio.mimwrite(track_video_path, cam_data['track_imgs'], fps=fps)

    #saving normal images
    for j in range(len(dataset["camtoworld"].keys())):
        outpath = f"{output_folder}/rendered_video_cam_{j}.mp4"
        imageio.mimwrite(outpath, (render_img_frames[:, j].squeeze()*255).astype(np.uint8), fps=fps)



    #Approach 1: good way
    opacity_threshold = 0.1
    viz_pc_t0 = gt_t0_all
    opacity_t0 = params["opacities"][0]
    gt_idxs_viz_pc = find_closest_gauss(viz_pc_t0, params["means"][0].cpu().numpy())
    opacities_idxs_viz_pc = (torch.sigmoid(opacity_t0))[gt_idxs_viz_pc].squeeze().cpu()
    gt_idxs_viz_pc = gt_idxs_viz_pc[opacities_idxs_viz_pc > opacity_threshold]
    visible_points = params["means"][:,gt_idxs_viz_pc, :3]
    torch.save(visible_points, f"{output_folder}/point_cloud_trajectory.pt")
    min_vals = np.min(visible_points.cpu().numpy(), axis=(0,1))
    max_vals = np.max(visible_points.cpu().numpy(), axis=(0,1))
    # center_position= (min_vals + max_vals) /2
   
    if "clematis" in data_dir:
        center_position = [0.00545695, -0.0413458 ,  1.680124]
    elif "rose" in data_dir:
        center_position = [-0.01537376, -0.02297388,  1.6785533]
    elif "lily" in data_dir:
        center_position = [-0.01201824, -0.00301804, 1.6874188]
    elif "tulip" in data_dir:
        center_position = [0.01096831, 0.00259373, 1.6566422] 
    elif "plant_1" in data_dir:
        center_position = [-0.01575137, -0.00203469,  1.6202013]
    elif "plant_2" in data_dir:
        center_position = [ 0.00193186, -0.00170395,  1.6401193 ]
    elif "plant_3" in data_dir:
        center_position = [6.3185021e-04, 5.7011396e-03, 1.6463835e+00]
    elif "plant_4" in data_dir:
        center_position = [ 0.01442758, -0.00233341,  1.6272888 ] 

    # elevation, azimuth = 15.732388496398926, -86.39990997314453
    # global_depth_min, global_depth_max, depth_reference_point = calculate_global_depth_range(visible_points, center_position, view_angles=None, min_vals=min_vals, max_vals=max_vals)
    #just compute depths once and use it for colors
    pose_dict = {"r_0": [15.732388496398926, -86.39990997314453],
                        "r_1": [15.698765754699707, 89.99995422363281], 
                        "r_2": [15.706961631774902, -82.79780578613281]}
    for i, pose in enumerate(pose_dict.items()):
        elevation, azimuth = pose[1]
        animate_point_clouds(
            visible_points,
            figsize=(6, 6),
            output_file=f"{output_folder}/point_cloud_animation.mp4",
            is_reverse=False,
            center_position=center_position,
            min_vals=min_vals,
            max_vals=max_vals,
            view_angles=(elevation, azimuth)
            # global_depth_min=global_depth_min,
            # global_depth_max=global_depth_max
        )
        #save individual point cloud frames 
        os.makedirs(f"{output_folder}/point_clouds/r_{i}", exist_ok=True)
        for j, point in enumerate(visible_points):
            visualize_point_cloud(
                point,
                figsize=(6, 6),
                output_file=f"{output_folder}/point_clouds/r_{i}/point_cloud_{j}.png",
                center_position=center_position,
                min_vals=min_vals,
                max_vals=max_vals,
                view_angles=(elevation,azimuth)
                # global_depth_min=global_depth_min,
                # global_depth_max=global_depth_max
            )


@torch.no_grad() 
def render_eval_interp(exp_path, data_dir, every_t, split, cached_params_path=None, render_white=False):
    """Render eval stuff"""
    md_path =  f"{data_dir}/transforms_{split}.json"
    with open(md_path, "r") as f:
        md = json.load(f)

    #use train_md to get the times
    train_path =  f"{data_dir}/transforms_train.json"
    with open(train_path, "r") as f:
        train_md = json.load(f)

    #collect all timesteps for the first view
    training_times = []
    for frame in train_md["frames"]:
        if frame["time"] == 1.0:
            training_times.append(frame["time"])
            break
        else:
            training_times.append(frame["time"])

    variables, images, cam_to_worlds_dict, intrinsics, alpha_masks = initialize_params_and_get_data(data_dir, md, every_t, render_white)
    param_path = glob(f"{exp_path}/*.npz")[0]
    param_dict = np.load(param_path) #each param is of shape (T, N, F)
    original_params = {}

    if render_white:
        output_folder = os.path.join(exp_path, split+"_white")
    else:
        output_folder = os.path.join(exp_path, split)
        os.makedirs(output_folder, exist_ok=True)

    for k,v in param_dict.items():
        original_params[k] = torch.tensor(v).to("cuda")

    N =  original_params["means"].shape[1]
    # trained_timesteps = len(training_times)
    gaussians_dict = {}
    for i,t in enumerate(training_times):
        gaussians_dict[t] = {
            "means": original_params["means"][i].cpu(),
            "quaternions": original_params["quats"][i].cpu(),
            "colors": original_params["rgbs"][i].cpu(),
        }
    interpolator = GaussianInterpolator(
        trained_timesteps=training_times,
        total_timesteps=70,
        degree=3  # Cubic interpolation
    )
    if cached_params_path:
        cached_params = torch.load(cached_params_path)
        params = cached_params
    else:
        cached_params = None 


    # Interpolate all parameters
    if not cached_params:
        all_gaussians = interpolator.interpolate_all(gaussians_dict)
        # Stack all timesteps (assuming they're sorted)
        sorted_timesteps = sorted(all_gaussians.keys())
        means = torch.stack([torch.from_numpy(all_gaussians[t]["means"]) for t in sorted_timesteps]).to("cuda")
        quats = torch.stack([torch.from_numpy(all_gaussians[t]["quaternions"]) for t in sorted_timesteps]).to("cuda")
        colors = torch.stack([torch.from_numpy(all_gaussians[t]["colors"]) for t in sorted_timesteps]).to("cuda")

        params = {
            "means": means.to(torch.float32),
            "quats": quats.to(torch.float32),
            "rgbs": colors.to(torch.float32),
            "scales": original_params["scales"][0:1].expand(70, -1, -1),
            "opacities": original_params["opacities"][0:1].expand(70, -1)
        }
        torch.save(params, os.path.join(exp_path, "gaussian_params.pt"))
    else:
        params = cached_params
    num_timesteps = params["means"].shape[0]
    render_img_frames = np.zeros((num_timesteps, len(images.keys()), 400, 400, 3))
    
    cam_indices_lst = list(images.keys())
    for cam_idx in cam_indices_lst:
        full_cam_path = os.path.join(output_folder, cam_idx)
        os.makedirs(full_cam_path, exist_ok=True)

    gt_tracks_path = None
    scene = None
    gt_idxs_viz_pc = None
    render_tracks = False if render_white else True
    if split == "test":
        # Determine scene name
        scene_names = ["plant_1", "plant_2", "plant_3", "plant_4", "plant_5", 
                      "rose", "lily", "tulip", "clematis", "peony"]
        for scene_name in scene_names:
            if scene_name in data_dir:
                scene = scene_name
                break
        
        gt_tracks_path = os.path.join(data_dir, "meshes", f"relevant_{scene}_meshes", "trajectory_frames.npz")
        if os.path.exists(gt_tracks_path):
            gt_tracks = np.load(gt_tracks_path)
            gt_t0_all = gt_tracks["frame_0000"]  # (N,3)
            
            # Setup for point cloud visualization

    #Create all folders/subfolders
    if render_tracks and gt_tracks_path and os.path.exists(gt_tracks_path):
        tracks_output_folder = os.path.join(output_folder, "tracks")
        os.makedirs(tracks_output_folder, exist_ok=True)
        
        # Track visualization parameters
        if "plant" not in scene:
            subsample_factor_tracks = 15 #prevent overcrowding (10k points is good)
        else:
            subsample_factor_tracks = 1 #the plants scene have fewer mesh vertices so no need to subsample
        gt_t0 = gt_t0_all[::subsample_factor_tracks] #make flows less crowded
        opacity_threshold_flow = 0.3
        tracking_window = 5
        arrow_thickness = 2
        flow_skip = 1
        show_only_visible = True
        fade_strength = 0.8
        min_alpha = 0.1
        
        print(f"GT tracks has {gt_t0.shape[0]} points")
        
        #Using the same gt_idxs
        camera_track_data = {}
        gt_idxs = find_closest_gauss(gt_t0, params["means"][0, :, :3].cpu().numpy())
        if show_only_visible:
            opacities_idxs = torch.sigmoid(params["opacities"][0, gt_idxs]).cpu()
            gt_idxs = gt_idxs[opacities_idxs > opacity_threshold_flow]
        n_gaussians = gt_idxs.shape[0]
        cmap = plt.cm.get_cmap("jet") #following tracking everything
        colors = []
        for i in range(n_gaussians):
            color = cmap(i / n_gaussians)[:3]  # Get RGB, ignore alpha
            colors.append((int(color[0]*255), int(color[1]*255), int(color[2]*255)))

        # Initialize tracking variables per camera
        for cam_idx in cam_indices_lst:
            camera_track_data[cam_idx] = {
                'gt_idxs': gt_idxs,
                'all_trajs': None,
                'all_times': None,
                'colors': colors,
                'track_imgs': [],
                'track_path': os.path.join(tracks_output_folder, cam_idx)
            }
            os.makedirs(camera_track_data[cam_idx]['track_path'], exist_ok=True)

    #main rendering loop
    for t in tqdm(range(num_timesteps)):
        dataset = get_dataset(t, images, cam_to_worlds_dict, intrinsics) #getting all cameras for time t
        time_params = {k: v[t] for k,v in params.items()}
        pred_images = render_imgs(dataset, time_params, variables, t)
        render_img_frames[t] = pred_images.cpu().numpy()
        for i, img in enumerate(pred_images): #loop over each camera
            full_cam_path = os.path.join(output_folder, cam_indices_lst[i])
            alpha_mask_i = alpha_masks[f"r_{i}"][t]
            full_image_path = os.path.join(full_cam_path, f"{t:05d}.png")
            if alpha_mask_i.ndim == 2:
                alpha_mask_i = alpha_mask_i.unsqueeze(-1)  # Add channel dimension
            if render_white:
                background = torch.ones_like(img)
            else:
                background = torch.zeros_like(img) 
            composited_img = img * alpha_mask_i + background* (1 - alpha_mask_i)
            imageio.imwrite(full_image_path, (composited_img.cpu().numpy()*255).astype(np.uint8))

        if render_tracks and gt_tracks_path and os.path.exists(gt_tracks_path):
            current_means3d = params["means"][t, :, :3]  # (N, 3)
            
            for i, cam_idx in enumerate(cam_indices_lst):
                cam_data = camera_track_data[cam_idx]
                
                # Get current rendering
                current_rendering = (pred_images[i].cpu().numpy() * 255).astype(np.uint8).copy()
                
                # Initialize gt_idxs for this camera if not done
                
                # Get current camera viewmat
                current_c2w = dataset["camtoworld"][cam_idx]
                current_viewmat = torch.linalg.inv(current_c2w.to("cuda"))
                
                # Project current 3D positions to 2D
                selected_means = current_means3d[cam_data['gt_idxs']]
                current_means_cam = world_to_cam_means(selected_means, current_viewmat[None])
                means_2d = pers_proj_means(current_means_cam, dataset["K"][None].to("cuda"), 400, 400)
                current_projections = means_2d.squeeze()
                
                # Update 3D trajectories
                if cam_data['all_trajs'] is None:
                    cam_data['all_times'] = np.array([t])
                    cam_data['all_trajs'] = selected_means.unsqueeze(0).cpu().numpy()
                else:
                    cam_data['all_times'] = np.concatenate((cam_data['all_times'], np.array([t])), axis=0)
                    cam_data['all_trajs'] = np.concatenate((cam_data['all_trajs'], selected_means.unsqueeze(0).cpu().numpy()), axis=0)
                
                # Create trajectory visualization
                current_projections_np = current_projections.cpu().numpy()
                n_gaussians = len(cam_data['gt_idxs'])
                
                # Mask for valid projections
                current_mask = (current_projections_np[:, 0] >= 0) & (current_projections_np[:, 0] < 400) & \
                              (current_projections_np[:, 1] >= 0) & (current_projections_np[:, 1] < 400)
                
                # Draw current points
                for idx in range(0, n_gaussians, flow_skip):
                    if current_mask[idx]:
                        color_idx = (idx // flow_skip) % len(cam_data['colors'])
                        cv2.circle(current_rendering, 
                                 (int(current_projections_np[idx, 0]), int(current_projections_np[idx, 1])), 
                                 2, cam_data['colors'][color_idx], -1)
                
                # Draw trajectories if we have multiple frames
                if cam_data['all_trajs'].shape[0] > 1:
                    traj_img = np.ascontiguousarray(np.zeros((400, 400, 3), dtype=np.uint8))
                    
                    # Apply tracking window
                    all_trajs = cam_data['all_trajs']
                    all_times = cam_data['all_times']
                    if tracking_window is not None and tracking_window < all_trajs.shape[0]:
                        all_trajs = all_trajs[-tracking_window:]
                        all_times = all_times[-tracking_window:]
                    
                    # Draw trajectory lines
                    for t_idx in range(all_trajs.shape[0] - 1):
                        prev_gaussians = torch.from_numpy(all_trajs[t_idx]).to("cuda")
                        prev_projections_cam = world_to_cam_means(prev_gaussians, current_viewmat[None])
                        prev_projections = pers_proj_means(prev_projections_cam, dataset["K"][None].to("cuda"), 400, 400)
                        prev_projections = prev_projections.squeeze()
                        prev_time = all_times[t_idx]
                        
                        curr_gaussians = torch.from_numpy(all_trajs[t_idx + 1]).to("cuda")
                        curr_projections_cam = world_to_cam_means(curr_gaussians, current_viewmat[None])
                        curr_projections = pers_proj_means(curr_projections_cam, dataset["K"][None].to("cuda"), 400, 400)
                        curr_projections = curr_projections.squeeze()
                        curr_time = all_times[t_idx + 1]
                        
                        # Calculate fade factor
                        time_diff = t - curr_time
                        max_time_diff = t - all_times[0] if len(all_times) > 1 else 1
                        
                        if max_time_diff > 0:
                            fade_factor = 1.0 - (time_diff / max_time_diff) * fade_strength
                            fade_factor = max(fade_factor, min_alpha)
                        else:
                            fade_factor = 1.0
                        
                        # Get masks for valid 2D projections
                        prev_projections_np = prev_projections.cpu().numpy()
                        curr_projections_np = curr_projections.cpu().numpy()
                        
                        prev_mask = (prev_projections_np[:, 0] >= 0) & (prev_projections_np[:, 0] < 400) & \
                                   (prev_projections_np[:, 1] >= 0) & (prev_projections_np[:, 1] < 400)
                        curr_mask = (curr_projections_np[:, 0] >= 0) & (curr_projections_np[:, 0] < 400) & \
                                   (curr_projections_np[:, 1] >= 0) & (curr_projections_np[:, 1] < 400)
                        
                        # Draw trajectory lines
                        if curr_time <= t and prev_time <= t:
                            for idx in range(0, curr_projections.shape[0], flow_skip):
                                color_idx = (idx // flow_skip) % len(cam_data['colors'])
                                if prev_mask[idx] and curr_mask[idx]:
                                    faded_color = tuple(int(c * fade_factor) for c in cam_data['colors'][color_idx])
                                    traj_img = cv2.line(traj_img,
                                                       (int(prev_projections_np[idx, 0]), int(prev_projections_np[idx, 1])),
                                                       (int(curr_projections_np[idx, 0]), int(curr_projections_np[idx, 1])),
                                                       faded_color, arrow_thickness)
                    
                    # Overlay trajectories on rendering
                    current_rendering[traj_img > 0] = traj_img[traj_img > 0]
                
                # Save track visualization
                track_image_path = os.path.join(cam_data['track_path'], f"{t:05d}.png")
                imageio.imwrite(track_image_path, current_rendering)
                cam_data['track_imgs'].append(current_rendering)
    #saving tracks
    fps = len(render_img_frames)/3
    if render_tracks and gt_tracks_path and os.path.exists(gt_tracks_path):
        for cam_idx in cam_indices_lst:
            cam_data = camera_track_data[cam_idx]
            if cam_data['track_imgs']:
                track_video_path = f"{cam_data['track_path']}/full_track.mp4"
                imageio.mimwrite(track_video_path, cam_data['track_imgs'], fps=fps)

    #saving normal images
    for j in range(len(dataset["camtoworld"].keys())):
        outpath = f"{output_folder}/rendered_video_cam_{j}.mp4"
        imageio.mimwrite(outpath, (render_img_frames[:, j].squeeze()*255).astype(np.uint8), fps=fps)

    #Approach 1: good way
    opacity_threshold = 0.1
    viz_pc_t0 = gt_t0_all
    opacity_t0 = params["opacities"][0]
    gt_idxs_viz_pc = find_closest_gauss(viz_pc_t0, params["means"][0].cpu().numpy())
    opacities_idxs_viz_pc = (torch.sigmoid(opacity_t0))[gt_idxs_viz_pc].squeeze().cpu()
    gt_idxs_viz_pc = gt_idxs_viz_pc[opacities_idxs_viz_pc > opacity_threshold]
    visible_points = params["means"][:,gt_idxs_viz_pc, :3]
    torch.save(visible_points, f"{output_folder}/point_cloud_trajectory.pt")
    min_vals = np.min(visible_points.cpu().numpy(), axis=(0,1))
    max_vals = np.max(visible_points.cpu().numpy(), axis=(0,1))
    # center_position= (min_vals + max_vals) /2
   
    if "clematis" in data_dir:
        center_position = [0.00250162, -0.0451958,1.6817672]
    elif "rose" in data_dir:
        center_position = [-0.01537376, -0.02297388,  1.6785533]
    elif "lily" in data_dir:
        center_position = [-0.01201824, -0.00301804, 1.6874188]
    elif "tulip" in data_dir:
        center_position = [0.0130202 , 0.00563216, 1.6561513] 
    elif "plant_1" in data_dir:
        center_position = [-1.24790855e-02, 6.82123005e-04, 1.60255575e+00]
    elif "plant_2" in data_dir:
        center_position = [ 2.4079531e-04, -6.9841929e-03,  1.6393759e+00]
    elif "plant_3" in data_dir:
        center_position = [-1.5169904e-03, 7.5232387e-03, 1.6430800e+00]
    elif "plant_4" in data_dir:
        center_position = [0.01340409, 0.00430154, 1.6087356]

    # elevation, azimuth = 15.732388496398926, -86.39990997314453
    # global_depth_min, global_depth_max, depth_reference_point = calculate_global_depth_range(visible_points, center_position, view_angles=None, min_vals=min_vals, max_vals=max_vals)
    #just compute depths once and use it for colors
    pose_dict = {"r_0": [15.732388496398926, -86.39990997314453],
                        "r_1": [15.698765754699707, 89.99995422363281], 
                        "r_2": [15.706961631774902, -82.79780578613281]}
    for i, pose in enumerate(pose_dict.items()):
        elevation, azimuth = pose[1]
        # colors_clipped = np.clip(colors_np, 0, 3)  # Adjust 3 based on what looks good

        animate_point_clouds(
            visible_points,
            figsize=(6, 6),
            output_file=f"{output_folder}/point_cloud_gs_color_animation_r_{i}.mp4",
            is_reverse=False,
            center_position=center_position,
            min_vals=min_vals,
            max_vals=max_vals,
            view_angles=(elevation, azimuth),
            use_z_coloring=False,
            color=torch.clamp(colors[:, gt_idxs_viz_pc], 0, 1).cpu().numpy()
            # global_depth_min=global_depth_min,
            # global_depth_max=global_depth_max
        )
        #save individual point cloud frames 
        os.makedirs(f"{output_folder}/point_clouds_gs_color/r_{i}", exist_ok=True)
        for j, point in enumerate(visible_points):
            visualize_point_cloud(
                point,
                figsize=(6, 6),
                output_file=f"{output_folder}/point_clouds_gs_color/r_{i}/point_cloud_{j}.png",
                center_position=center_position,
                min_vals=min_vals,
                max_vals=max_vals,
                view_angles=(elevation,azimuth),
                use_z_coloring=False,
                color=torch.clamp(colors[j, gt_idxs_viz_pc], 0, 1).cpu().numpy()
                # global_depth_min=global_depth_min,
                # global_depth_max=global_depth_max
            )

            

if __name__ == "__main__":
    """Use this script to render images in a standard format so we can run metrics"""
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--data_dir", "-d", default="")
    parser.add_argument("--exp_path", "-p", default="")
    parser.add_argument("--every_t", "-t", type=int, default=1) #for eval need to match the number of point clouds saved
    parser.add_argument("--split", nargs="+", default=["test"]) #dont render train for now.
    parser.add_argument("--render_interp", action="store_true")
    parser.add_argument("--cached_params_path", type=str)
    parser.add_argument("--render_white", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir
    every_t = args.every_t
    exp_path = args.exp_path
    # for split in args.split:
    #     print(f"rendering {split}")
    #     render_eval(exp_path, data_dir, every_t, split)
    if args.render_interp:
        render_eval_interp(exp_path, data_dir, every_t, "test", args.cached_params_path, args.render_white)
    else:
        render_eval(exp_path, data_dir, every_t, "test")
    print("Done eval")
    torch.cuda.empty_cache()

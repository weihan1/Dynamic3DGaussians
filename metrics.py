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
    o3d_knn, params2rendervar, params2cpu, save_params

from external import calc_psnr, build_rotation, densify, update_params_and_optimizer
from argparse import ArgumentParser
import imageio.v2 as imageio
from helpers import knn

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def visualize_point_cloud(point_clouds, output_file="point_cloud.png", subsample_factor=1, 
                          point_size=20, figsize=(10, 8), view_angles=None, color='blue', zoom_factor=0.5):
    """
    Visualize a single-time point clouds (N, 3)
    This can be helpful for visualizing point cloud trajectory
    """
    if isinstance(point_clouds, torch.Tensor):
        point_clouds = point_clouds.cpu().numpy()

    point_clouds = point_clouds[::subsample_factor,:]
    N, _ = point_clouds.shape
    min_vals = point_clouds.min(axis=0)
    max_vals = point_clouds.max(axis=0)
    center = (min_vals + max_vals) / 2
    max_range = max(max_vals - min_vals)
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    x = point_clouds[:, 0]
    y = point_clouds[:, 1]
    z = point_clouds[:, 2]
    
    # Create scatter plot
    scatter = ax.scatter(x, y, z, s=point_size, c=color, alpha=0.8)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title
    ax.set_title(f'Point Cloud Visualization ({N} points)')
    
    # Set view angle if specified
    if view_angles is not None:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    # Set zoomed-in axis limits based on min/max and zoom factor
    ax.set_xlim([center[0] - max_range * zoom_factor, center[0] + max_range * zoom_factor])
    ax.set_ylim([center[1] - max_range * zoom_factor, center[1] + max_range * zoom_factor])
    ax.set_zlim([center[2] - max_range * zoom_factor, center[2] + max_range * zoom_factor])
    
    # Add coordinate axes for orientation reference
    max_length = max_range * 0.2  # Length of the coordinate axes lines
    origin = center - max_range * 0.4  # Offset origin point
    
    # X axis - red
    ax.plot([origin[0], origin[0] + max_length], [origin[1], origin[1]], [origin[2], origin[2]], 'r-', linewidth=2)
    ax.text(origin[0] + max_length * 1.1, origin[1], origin[2], 'X', color='red')
    
    # Y axis - green
    ax.plot([origin[0], origin[0]], [origin[1], origin[1] + max_length], [origin[2], origin[2]], 'g-', linewidth=2)
    ax.text(origin[0], origin[1] + max_length * 1.1, origin[2], 'Y', color='green')
    
    # Z axis - blue
    ax.plot([origin[0], origin[0]], [origin[1], origin[1]], [origin[2], origin[2] + max_length], 'b-', linewidth=2)
    ax.text(origin[0], origin[1], origin[2] + max_length * 1.1, 'Z', color='blue')
    
    # Add rotation indicators
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300)
    print(f"Visualization saved to {output_file}")
    
    # Show the plot
    plt.show()
    
    return fig, ax

def animate_point_clouds(point_clouds, output_file="point_cloud_animation.mp4", fps=10, 
                         point_size=20, figsize=(10, 8), view_angles=None, color='blue', is_reverse=True,
                         t_subsample=1):
    """
    Animate a sequence of point clouds and save as a video.
    
    Parameters:
    -----------
    point_clouds : numpy.ndarray
        Point clouds of shape (T, N, 3) where:
        - T is the number of frames
        - N is the number of points
        - 3 represents the XYZ coordinates
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
    if isinstance(point_clouds, torch.Tensor):
        point_clouds = point_clouds.cpu().numpy()
    every_n = int(1/t_subsample)
    point_clouds = point_clouds[::every_n,:,:]
    # Get data dimensions
    T, N, _ = point_clouds.shape
    
    if is_reverse:
        point_clouds = np.flip(point_clouds, axis=0)
     
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
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    # Set title
    ax.set_title('Point Cloud Animation')
    
    # Set initial view angle if specified
    if view_angles is not None:
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    def update(frame):
        """ Update function for each frame """
        x, y, z = point_clouds[frame, :, 0], point_clouds[frame, :, 1], point_clouds[frame, :, 2]
        
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


def get_batch(dataset):
    """
    Copies per-time dataset onto todo_dataset and randomly selects a camera from that timestep
    TODO: for now, just implement their method by randomly selecting a camera 
    TODO: one extension would be to sample all cameras and see if that improves the performance
    """
    curr_data = {"Ks": dataset["K"].unsqueeze(0).to("cuda"),
                 "width": dataset["width"],
                 "height": dataset["height"]}

    all_camera_indices = list(dataset["camtoworld"])
    selected_camera_ind = random.choice(all_camera_indices)
    selected_c2w = dataset["camtoworld"][selected_camera_ind]
    selected_image = dataset["image"][selected_camera_ind]
    curr_data["c2w"] = selected_c2w.to("cuda")
    curr_data["image"] = selected_image.to("cuda") 
    return curr_data


def initialize_params_and_get_data(data_dir, md, every_t, is_reverse=True, num_gaussians=100_000):
    """
    Use this when training on custom plant dataset.
    It has "camera_angle_x" and "frames" as keys
    1. Load images
    2. Initialize gaussian parameters
    """
    cam_to_worlds_dict = {} #we just need on cam to worlds per view
    images = {} #(N: T, H, W, C)
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
        if camera_id not in images:
            images[camera_id] = []
            images[camera_id].append(norm_img)
            image_ids_dict[camera_id] = []
            image_ids_dict[camera_id].append(image_ids)
        else:
            images[camera_id].append(norm_img)
            image_ids_dict[camera_id].append(image_ids)

        c2w = torch.tensor(frame["transform_matrix"])
        c2w[0:3, 1:3] *= -1  # Convert from OpenGL to OpenCV

        cam_to_worlds_dict[camera_id] = c2w
        progress_bar.update(1)
        images_ids_lst.append(image_ids)

    for camera_id in images.keys(): #subsample the image list for each camera, btw this downsamples in both splits
        images[camera_id] = images[camera_id][::-every_t] if is_reverse else images[camera_id][::every_t]
        image_ids_dict[camera_id] = image_ids_dict[camera_id][::-every_t] if is_reverse else image_ids_dict[camera_id][::every_t]

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
    # init_extent = 0.5
    # init_scale = 1.0
    # init_opacity = 0.1
    
    #initialize the parameters
    # points = init_extent * scene_scale * (torch.rand((num_gaussians, 3)) * 2 - 1)
    # print(f"points are initialized in [{points.min(), points.max()}]")
    # rgbs = torch.rand((num_gaussians, 3))
    # dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    # dist_avg = torch.sqrt(dist2_avg)
    # scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    # N = points.shape[0]
    # quats = torch.rand((N, 4))  # [N, 4]
    # opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    # params = {
    #     'means': points,
    #     'rgbs': rgbs,
    #     'quats':  quats, 
    #     'opacities': opacities,
    #     'scales': scales,
    # }

    # params = {k: torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True)) for k, v in
    #           params.items()}

    
    variables = {'scene_scale': scene_scale,
                 'fixed_bkgd': torch.zeros(1, 3, device="cuda")} #fixed black background we use for rendering eval for instance.

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


def initialize_optimizer(params, variables):
    lrs = {
        'means': 0.00016 * variables["scene_scale"],
        'rgbs': 0.0025,
        'quats': 0.001,
        'opacities': 0.05, 
        'scales': 0.005 #changed from 0.001
    }
    # param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    optimizer_class = torch.optim.Adam
    optimizer = {
        name: optimizer_class(
            [{"params": params[name], "lr": lr, "name": name}],
            eps=1e-15, 
        )
        for name, lr in lrs.items()
    }
    return optimizer

def initialize_per_timestep(params, variables, optimizer):
    pts = params['means']
    #  forward estimate based on a velocity 
    rot = torch.nn.functional.normalize(params['quats'])
    new_pts = pts + (pts - variables["prev_pts"])
    new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    # is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts
    #this prev offset calculates the distance between each point and its neighbor. 
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None] #distance between each point and all of its neighbors
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgbs'].detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()

    new_params = {'means': new_pts, 'quats': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    """Calculating the neighbors of each gaussians means and freezing opacities and scales"""
    # is_fg = params['seg_colors'][:, 0] > 0.5
    # init_fg_pts = params['means3D'][is_fg]
    # init_bg_pts = params['means3D'][~is_fg]
    # init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    init_fg_pts = params["means"]
    #1. calculates the distances and weights for each gaussian and its neighbors.
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn) #excludes itself as neighbor
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist) #(N, num_knn)
    neighbor_dist = np.sqrt(neighbor_sq_dist) #(N, num_knn)

    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    # variables["init_bg_pts"] = init_bg_pts.detach()
    # variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params["means"].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params["quats"]).detach()
    params_to_fix = ['opacities', 'scales']
    for param_name, opt_class in optimizer.items():
        if param_name in params_to_fix:
            opt_class.param_groups[0]["lr"] = 0.0
    return variables


def report_progress(params, variables, dataset, i, progress_bar, cam_ind=0, every_i=100, use_random_bkgd=True):
    """
    Report progress on the camera of cam_ind. Using the same post-processing scheme from training.
    NOTE: record masked psnr.
    """
    if i % every_i == 0:
        all_camera_indices = list(dataset["camtoworld"])
        selected_camera = all_camera_indices[cam_ind]
        selected_c2w = dataset["camtoworld"][selected_camera].to("cuda")
        rendervar = params2rendervar(params)
        means = rendervar["means"]
        colors = rendervar["colors"]
        quats = rendervar["quats"]
        opacities = rendervar["opacities"]
        scales = rendervar["scales"]
        im, alphas, info = rasterize_splats(means, quats,scales,opacities, colors,
                                                camtoworlds=selected_c2w[None], Ks=dataset["K"].unsqueeze(0).to("cuda"),
                                                width=dataset["width"], height=dataset["height"])
        pred_image = im[0]
        gt_image = dataset["image"][selected_camera].to("cuda")
        gt_has_alpha = gt_image.shape[-1] == 4

        #NOTE: doesn't matter what bkgd we use cause we are evaluating masked psnr.
        bkgd = variables["fixed_bkgd"]

        if gt_has_alpha:
            gt_alpha = gt_image[..., [-1]]
            mask = (gt_alpha > 0).squeeze() #do not use a hardcap at 1
            gt_rgb = gt_image[..., :3]
            pixels = gt_rgb * gt_alpha + bkgd * (1 - gt_alpha)
            pred_image = pred_image + bkgd * (1.0 - alphas)
        elif use_random_bkgd:
            colors = colors + bkgd * (1.0 - alphas)

        pred_image = torch.clamp(pred_image, 0, 1).squeeze()
        gt_image = torch.clamp(pixels, 0, 1).squeeze() 
        psnr = calc_psnr(pred_image, gt_image, mask)
        progress_bar.set_postfix({f"train img {cam_ind} mask PSNR": f"{psnr:.{2}f}",
                                  "num_gauss": means.shape[0]})
        progress_bar.update(every_i)

def save_gt_video(video_duration, video, cam_index, full_out_path, is_reverse=True):
    """
    Given an array of frames, and cam index, save that video to full_out_path
    """
    #no need to background blend
    selected_video = video[cam_index]
    selected_video = np.stack(selected_video, axis=0)
    fps = len(selected_video)/video_duration
    if is_reverse:
        selected_video = np.flip(selected_video, axis=0)
    imageio.mimwrite(full_out_path, (selected_video*255).astype(np.uint8), fps=fps)

    

@torch.no_grad()
def render_imgs(dataset, params, variables, val_path, timestep, iteration):
    """
    Render a canvas with pred and gt images.
    Given dataset, render images from all possible views of this dataset.
    Compute average masked PSNR over all training cameras across all views.
    """
    curr_data = {"Ks": dataset["K"].unsqueeze(0).to("cuda"),
                 "width": dataset["width"],
                 "height": dataset["height"]}

    all_camera_indices = list(dataset["camtoworld"])

    calc_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    calc_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda") #. If set to True will instead expect input to be in the [0,1] range.

    # elapsed_time = 0
    metrics = defaultdict(list)
    float_metrics = defaultdict(list)
    num_cameras = len(all_camera_indices)
    h,w = dataset["image"]["r_0"].shape[0], dataset["image"]["r_0"].shape[1]

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

        canvas_list = [image_pixels, image_colors] #for display, don't do alpha blending.

        # write images
        canvas = torch.cat(canvas_list, dim=1).squeeze(0).cpu().numpy()
        canvas = (canvas * 255).astype(np.uint8)
        imageio.imwrite(
            f"{val_path}/t_{timestep}_eval_{camera_indx}.png",
            canvas,
        )

        #Compute masked psnr 
        pred_image = eval_colors.squeeze() #(H,W,3)
        pred_images[i] = pred_image
        gt_image = eval_pixels
        if gt_has_alpha:
            mask = (gt_alpha > 0).squeeze()
            psnr_value = calc_psnr(pred_image, gt_image, mask)
        else:
            psnr_value = calc_psnr(pred_image, gt_image)

        _ssim = calc_ssim(eval_colors.permute(0,3,1,2), eval_pixels.unsqueeze(0).permute(0,3,1,2))
        _lpips = calc_lpips(eval_colors.permute(0,3,1,2), eval_pixels.unsqueeze(0).permute(0,3,1,2))
        metrics["psnr"].append(psnr_value)
        metrics["ssim"].append(_ssim)
        metrics["lpips"].append(_lpips)

        #Use float_metrics to store the float values as opposed to torch tensors
        float_metrics["psnr"].append(round(psnr_value.item(), 2))
        float_metrics["ssim"].append(round(_ssim.item(), 3))
        float_metrics["lpips"].append(round(_lpips.item(), 3))

    
    #metrics keep a dict of metrics where each metric is a list of values
    
    return metrics, float_metrics, pred_images 

@torch.no_grad() 
def render_eval(exp_path, data_dir, every_t):
    """Render eval stuff"""
    #1. Open transforms_test.json
    #2. Load test images     params, variables, images, cam_to_worlds_dict, intrinsics = initialize_params_and_get_data(data_dir, md, every_t)
    #3. Load the gaussian parameters 
    #4. Render predicted vs gt on the test pose
    #5. Loop over the number of timesteps and the number of test poses and compute psnr
    #6. save videos
    #NOTE: Recall in dynamic3DGS, we freeze everything except ['opacities', 'scales']

    md_test = json.load(open(f"{data_dir}/transforms_test.json", 'r'))
    variables, images, cam_to_worlds_dict, intrinsics = initialize_params_and_get_data(data_dir, md_test, every_t)
    param_path = glob(f"{exp_path}/*.npz")[0]
    param_dict = np.load(param_path) #each param is of shape (T, N, F)
    params = {}
    timestep_stats = {}
    all_psnr_values = []
    all_ssim_values = []
    all_lpips_values = []

    val_path = os.path.join(exp_path, "eval")
    os.makedirs(val_path, exist_ok=True)

    for k,v in param_dict.items():
        params[k] = torch.tensor(v).to("cuda")
    num_timesteps = params["means"].shape[0]
    render_img_frames = np.zeros((num_timesteps, len(images.keys()), 400, 400, 3))
    for t in range(num_timesteps):  
        dataset = get_dataset(t, images, cam_to_worlds_dict, intrinsics) #getting all cameras for time t
        for cam_index in dataset["camtoworld"].keys():
            save_gt_video(video_duration=3, video=images, cam_index=cam_index, full_out_path=f"{val_path}/gt_video_cam_{cam_index}.mp4", is_reverse=True)

        time_params = {k: v[t] for k,v in params.items()}
        num_cams = len(dataset["image"])
        cam_indices = list(dataset["image"])
        metrics, float_metrics, pred_image = render_imgs(dataset, time_params, variables, val_path, t, iteration=0)
        render_img_frames[t] = pred_image.cpu().numpy()

        #collect all metrics
        all_psnr_values.extend(metrics["psnr"])  # Collect all PSNR values
        all_ssim_values.extend(metrics["ssim"])
        all_lpips_values.extend(metrics["lpips"])

        timestep_stats[t] = {
            "per_camera_psnr": float_metrics["psnr"],
            "per_camera_ssim": float_metrics["ssim"],
            "per_camera_lpips": float_metrics["lpips"]
            }

    fps = len(render_img_frames)/3
    render_img_frames = np.flip(render_img_frames, axis=0)
    for i in range(len(dataset["camtoworld"].keys())):
        outpath = f"{val_path}/rendered_video_cam_{i}.mp4"
        imageio.mimwrite(outpath, (render_img_frames[:, i].squeeze()*255).astype(np.uint8), fps=fps)

    overall_mean_psnr = torch.stack(all_psnr_values).mean().item()
    overall_mean_ssim = torch.stack(all_ssim_values).mean().item()
    overall_mean_lpips = torch.stack(all_lpips_values).mean().item()

    stats = {
        "overall_mean_psnr": overall_mean_psnr,
        "overal_mean_ssim": overall_mean_ssim,
        "overall_mean_lpips": overall_mean_lpips,
        "timestep_stats": timestep_stats
        }
    with open(f"{val_path}/eval_metrics.json", "w") as f:
        json.dump(stats, f)

    threshold = 0.3 
    opacity_t0 = params["opacities"][0] #take opacity at t=0 for pruning 
    above_threshold_mask = (torch.sigmoid(opacity_t0) > threshold) #(N)
    visible_points = params["means"][:, above_threshold_mask, :]
    torch.save(visible_points, f"{exp_path}/point_cloud_trajectory.pt")
    # visualize_point_cloud(visible_points[0])
    animate_point_clouds(visible_points.cpu(), output_file=f"{exp_path}/point_cloud_animation.mp4",
                            is_reverse=True, t_subsample=1)

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--data_dir", "-d", default="./plant_data/rose")
    parser.add_argument("--exp_path", "-p", default="./output/exp1/rose_all_cams_15_timesteps")
    parser.add_argument("--every_t", "-t", type=int, default=10) #for eval need to match the number of point clouds saved
    args = parser.parse_args()
    data_dir = args.data_dir
    every_t = args.every_t
    exp_path = args.exp_path
    render_eval(exp_path, data_dir, every_t)
    print("Done eval")
    torch.cuda.empty_cache()

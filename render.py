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
from scipy.spatial import ConvexHull

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

        #Compute masked psnr 
        pred_image = eval_colors.squeeze() #(H,W,3)
        pred_images[i] = pred_image
    
    return pred_images 





@torch.no_grad() 
def render_eval(exp_path, data_dir, every_t, is_reverse):
    """Render eval stuff"""
    #NOTE: Recall in dynamic3DGS, we freeze everything except ['opacities', 'scales']
    md_test = json.load(open(f"{data_dir}/transforms_test.json", 'r'))
    variables, images, cam_to_worlds_dict, intrinsics = initialize_params_and_get_data(data_dir, md_test, every_t, is_reverse=is_reverse)
    param_path = glob(f"{exp_path}/*.npz")[0]
    param_dict = np.load(param_path) #each param is of shape (T, N, F)
    params = {}

    #NOTE: we are only going to do it for the test folder and not the train folder nor the gt
    test_path = os.path.join(exp_path, "test")
    os.makedirs(test_path, exist_ok=True)

    for k,v in param_dict.items():
        params[k] = torch.tensor(v).to("cuda")
    num_timesteps = params["means"].shape[0]
    render_img_frames = np.zeros((num_timesteps, len(images.keys()), 400, 400, 3))
    #Need to create the test camera folders
    cam_indices_lst = list(images.keys())
    for cam_idx in cam_indices_lst:
        full_cam_path = os.path.join(test_path, cam_idx)
        os.makedirs(full_cam_path, exist_ok=True)

    for t in range(num_timesteps):  
        dataset = get_dataset(t, images, cam_to_worlds_dict, intrinsics) #getting all cameras for time t
        time_params = {k: v[t] for k,v in params.items()}
        pred_images = render_imgs(dataset, time_params, variables, t)
        render_img_frames[t] = pred_images.cpu().numpy()
        for i, img in enumerate(pred_images):
            full_cam_path = os.path.join(test_path, cam_indices_lst[i])
            full_image_path = os.path.join(full_cam_path, f"{i:05d}.png")
            imageio.imwrite(full_image_path, (img.cpu().numpy()*255).astype(np.uint8))

    fps = len(render_img_frames)/3
    render_img_frames = np.flip(render_img_frames, axis=0)
    for j in range(len(dataset["camtoworld"].keys())):
        outpath = f"{test_path}/rendered_video_cam_{j}.mp4"
        imageio.mimwrite(outpath, (render_img_frames[:, j].squeeze()*255).astype(np.uint8), fps=fps)


if __name__ == "__main__":
    """Use this script to render images in a standard format so we can run metrics"""
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--data_dir", "-d", default="./plant_data/rose_transparent")
    parser.add_argument("--exp_path", "-p", default="./output/exp1/rose_baseline")
    parser.add_argument("--every_t", "-t", type=int, default=1) #for eval need to match the number of point clouds saved
    parser.add_argument("--is_reverse", "-r", type=bool, default=False) #NOTE: the reason we set is reverse equal to False here is so we can have the normal growth of plant in rendered
    args = parser.parse_args()
    data_dir = args.data_dir
    every_t = args.every_t
    exp_path = args.exp_path
    is_reverse = args.is_reverse
    render_eval(exp_path, data_dir, every_t, is_reverse)
    print("Done eval")
    torch.cuda.empty_cache()

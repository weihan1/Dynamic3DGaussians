"""
Use this script to reproduce the dynamic 3DGS baseline on our custom plant dataset.
Notice for our dataset we do not
"""
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

# Use our own rasterizer for fair comparison
# from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from gsplat import rasterization, DefaultStrategy
from helpers import l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
from argparse import ArgumentParser
import imageio.v2 as imageio
from helpers import knn

"""
TODO: Since i'm training last timestep on 27 cameras and middle timesteps on 9, modify the code to allow for that
like allow for specific indexing of cameras
"""
def get_dataset(t, images, cam_to_worlds_dict, intrinsics, selected_cam_ids = ["r_1", "r_2", "r_3", "r_4", 
                                                                                                "r_6", "r_7", "r_8", "r_9", "r_10"]):
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
        if t == 0 or camera_id in selected_cam_ids:
            data["camtoworld"][camera_id] = cam_to_worlds_dict[camera_id]
            data["image"][camera_id] = torch.from_numpy(img_list[t]).float()
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
    init_extent = 0.5
    init_scale = 1.0
    init_opacity = 0.1
    
    #initialize the parameters
    points = init_extent * scene_scale * (torch.rand((num_gaussians, 3)) * 2 - 1)
    print(f"points are initialized in [{points.min(), points.max()}]")
    rgbs = torch.rand((num_gaussians, 3))
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

    params = {k: torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}

    
    variables = {'scene_scale': scene_scale,
                 'fixed_bkgd': torch.zeros(1, 3, device="cuda")} #fixed black background we use for rendering eval for instance.

    return params, variables, images, cam_to_worlds_dict, intrinsics


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


def get_loss(params, curr_data, variables, is_initial_timestep, optimizer, strategy, strategy_state, i, use_random_bkgd = True):
    """
    Rasterize image and compute loss
    If use_random_bkgd is true, train our method by blending it with a random background at each training iteration.
    Else, do not blend.
    """
    losses = {}
    rendervar = params2rendervar(params)
    means = rendervar["means"]
    colors = rendervar["colors"]
    quats = rendervar["quats"]
    opacities = rendervar["opacities"]
    scales = rendervar["scales"]
    im, alphas, info = rasterize_splats(means, quats, scales, opacities, colors,
                                            camtoworlds=curr_data["c2w"][None], Ks=curr_data["Ks"],
                                            width=curr_data["width"], height=curr_data["height"])
    if is_initial_timestep:
        strategy.step_pre_backward(params=params, optimizers=optimizer, 
                                   state=strategy_state, step=i, info=info) #retains gradients for 2d means

    gt_image = curr_data["image"]
    fixed_bkgd = variables["fixed_bkgd"]
    gt_has_alpha = gt_image.shape[-1] == 4
    if use_random_bkgd:
        bkgd = torch.rand(1, 3, device="cuda") #this blends a new bkgd at every iteration
    else:
        bkgd = fixed_bkgd

    if gt_has_alpha:
        gt_alpha = gt_image[..., [-1]]
        gt_rgb = gt_image[..., :3]
        pixels = gt_rgb * gt_alpha + bkgd * (1 - gt_alpha)
        im = im + bkgd * (1.0 - alphas)
    elif use_random_bkgd:
        colors = colors + bkgd * (1.0 - alphas)

    losses['im'] = 0.8 * l1_loss_v1(im, pixels) + 0.2 * (1.0 - calc_ssim(im, pixels))

    if not is_initial_timestep:
        # is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means']
        fg_rot = rendervar['quats']

        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"]) #quaternion multiplication q_{t} * q_{t-1}^-1
        rot = build_rotation(rel_rot) #build rotation from quaternion
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
        losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                              variables["neighbor_weight"])

        losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
                                            variables["neighbor_weight"])

        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

        #TODO: enable these losses after, first use rigid, rot and iso
        # losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()

        # bg_pts = rendervar['means3D'][~is_fg]
        # bg_rot = rendervar['rotations'][~is_fg]
        # losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

        # losses['soft_col_cons'] = l1_loss_v2(params['rgbs'], variables["prev_col"])

    loss_weights = {'im': 1.0, 'seg': 3.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 20.0,
                    'soft_col_cons': 0.01}
    loss = sum([loss_weights[k] * v for k, v in losses.items()])

    return loss, variables, info


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

@torch.no_grad()
def render_imgs(dataset, params, variables, exp_path, timestep, iteration):
    """
    Render a canvas with pred and gt images.
    """
    curr_data = {"Ks": dataset["K"].unsqueeze(0).to("cuda"),
                 "width": dataset["width"],
                 "height": dataset["height"]}

    all_camera_indices = list(dataset["camtoworld"])

    # elapsed_time = 0
    metrics = defaultdict(list)
    for camera_indx in all_camera_indices:
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
            f"{exp_path}/t_{timestep}_iter_{iteration}.png",
            canvas,
        )

        #Compute masked psnr 
        pred_image = eval_colors.squeeze()
        gt_image = eval_pixels.squeeze()
        if gt_has_alpha:
            mask = (gt_alpha > 0).squeeze()
            metrics["psnr"].append(calc_psnr(pred_image, gt_image, mask))
        else:
            metrics["psnr"].append(calc_psnr(pred_image, gt_image))

        #TODO: fix these after wednesday's presentation
        # pixels_p = eval_pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
        # colors_p = eval_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
        # metrics["ssim"].append(calc_ssim(colors_p, pixels_p))
        # metrics["lpips"].append(calc_lpips(colors_p, pixels_p))

    # elapsed_time /= len(valloader)

    stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
    stats.update(
        {
            # "elapsed_time": elapsed_time,
            "num_GS": params["means"].shape[0],
        }
    )
    print(f"The PSNR at time {timestep}, iteration {iteration}: {stats['psnr']:.3f}")
    # print(
    #     f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
    #     f"Number of GS: {stats['num_GS']}"
    # )
    # save stats as json
    with open(f"{exp_path}/timestep_{timestep}_iter_{iteration}.json", "w") as f:
        json.dump(stats, f)
    

def train(data_dir, exp, every_t):
    """Training script for the rose scene, specifically tailored from my custom dataset."""
    #TODO: write rendering results here 
    scene_name = data_dir.split("/")[-1]
    now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    exp_path = f"./output/{exp}/{scene_name}_{now}"
    os.makedirs(exp_path, exist_ok=True)
    md = json.load(open(f"{data_dir}/transforms_train.json", 'r'))
    plot_intervals = [1, 2999, 6999, 9999] 
    strategy = DefaultStrategy()
    strategy.refine_stop_iter = 5000 #original code base only prunes before iteration 5000
    params, variables, images, cam_to_worlds_dict, intrinsics = initialize_params_and_get_data(data_dir, md, every_t)
    timesteps = list(range(0, len(images)))
    num_timesteps = len(timesteps)
    optimizer = initialize_optimizer(params, variables)

    # Do the strategy stuff here
    strategy.check_sanity(params, optimizer)
    strategy_state = strategy.initialize_state() #initializes running state
    output_params = [] #contain the gaussians params for each timestep.
    for t in timesteps: #training on timestep t
        dataset = get_dataset(t, images, cam_to_worlds_dict, intrinsics) #getting all cameras for time t
        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = 10_000 if is_initial_timestep else 2000
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            curr_data = get_batch(dataset) #randomly selects a camera and rasterize.
            loss, variables, info = get_loss(params, curr_data, variables, is_initial_timestep, optimizer, strategy, strategy_state, i)
            loss.backward()
            if is_initial_timestep:
                strategy.step_post_backward(params, optimizer, strategy_state, i, info) #growing and pruning
            with torch.no_grad():
                report_progress(params, variables, dataset, i, progress_bar)
                for opt in optimizer.values():
                    opt.step()
                    opt.zero_grad(set_to_none=True)

            if i in plot_intervals: 
                render_imgs(dataset, params, variables, exp_path, t, i)       

        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)

    #TODO: Need to write the NVS eval
    # render_eval
    save_params(output_params, exp_path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--data_dir", "-d", default="/projects/4DTimelapse/Dynamic3DGaussiansBaseline/plant_data/rose_mini")
    parser.add_argument("--name", "-n", type=str, default="exp1")
    parser.add_argument("--every_t", "-t", type=int, default=10)
    args = parser.parse_args()
    data_dir = args.data_dir
    exp_name = args.name
    every_t = args.every_t
    train(data_dir, exp_name, every_t)
    torch.cuda.empty_cache()

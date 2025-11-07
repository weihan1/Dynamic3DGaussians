import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
from helpers import  l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
from argparse import ArgumentParser
from gsplat import rasterization, DefaultStrategy
from datetime import datetime
from torch.utils.data import DataLoader 
from collections import defaultdict
from torch import Tensor
from typing import Optional, Tuple, Dict
import imageio.v2 as imageio

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def generate_ellipse_path_z(
    poses: np.ndarray,
    n_frames: int = 120,
    # const_speed: bool = True,
    variation: float = 0.0,
    phase: float = 0.0,
    height: float = 0.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=height (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], height])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                variation
                * (
                    z_low[2]
                    + (z_high - z_low)[2]
                    * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)
                )
                + height,
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # if const_speed:
    #     # Resample theta angles so that the velocity is closer to constant.
    #     lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    #     theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    #     positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(center - p, up, p) for p in positions])

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
) -> Tuple[Tensor, Tensor, Dict]:
    """
    Rasterize the splats, assume they are already activated
    module with the color of the splat.
    Also notice here we use near plane and far plane ONLY for blender scene!

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
        near_plane=0.01,
        far_plane=1e10
    )
    if masks is not None:
        render_colors[~masks] = 0
    return render_colors, render_alphas, info

@torch.no_grad()
def render_imgs_captured(testset,params, variables, exp_path, timestep, iteration):
    """
    Render a canvas with pred and gt images.
    Only render the first training image
    Re-create a new curr_data (dont mix with that of training)
    """
    curr_data = {}
    first_camera_indx = list(testset.timestep_poses[timestep].keys())[0]
    train_path = os.path.join(exp_path, "train")
    os.makedirs(train_path, exist_ok=True)
    # elapsed_time = 0
    metrics = defaultdict(list)
    selected_c2w = torch.from_numpy(testset.timestep_poses[timestep][first_camera_indx]).to(torch.float32)
    selected_image = torch.from_numpy(testset.timestep_images[timestep][first_camera_indx]).to(torch.float32)
    selected_intrinsics = torch.from_numpy(testset.timestep_intrinsics[timestep][1]).to(torch.float32)
    curr_data["Ks"] = selected_intrinsics.to("cuda")
    curr_data["c2w"] = selected_c2w.to("cuda")
    curr_data["image"] = selected_image.to("cuda") 
    width = selected_image.shape[1]
    height = selected_image.shape[0]
    curr_data["width"] = width
    curr_data["height"] = height
    rendervar = params2rendervar(params)
    means = rendervar["means"]
    colors = rendervar["colors"]
    quats = rendervar["quats"]
    opacities = rendervar["opacities"]
    scales = rendervar["scales"]
    im, alphas, info = rasterize_splats(means, quats, scales, opacities, colors,
                                            camtoworlds=curr_data["c2w"][None], Ks=curr_data["Ks"][None],
                                            width=curr_data["width"], height=curr_data["height"])
    gt_image = curr_data["image"]
    image_pixels = gt_image 
    im = torch.clamp(im, 0.0, 1.0)
    image_colors = im.squeeze() 

    canvas_list = [image_pixels, image_colors] #for display, don't do alpha blending.

    # write images
    canvas = torch.cat(canvas_list, dim=1).squeeze(0).cpu().numpy()
    canvas = (canvas * 255).astype(np.uint8)
    imageio.imwrite(
        f"{train_path}/t_{timestep}_iter_{iteration}_cam{first_camera_indx}.png",
        canvas,
    )
    #render masks if timestep == 0
    if timestep == 0:
        gt_masks = torch.from_numpy(testset.timestep_masks[timestep][first_camera_indx][...,None]).expand(-1, -1, 3).to(torch.float32).to("cuda")
        masks = params["masks"]
        pred_masks,_,_ = rasterize_splats(means, quats, scales, opacities, torch.sigmoid(masks.expand(-1, 3)),
                                                camtoworlds=curr_data["c2w"][None], Ks=curr_data["Ks"][None],
                                                width=curr_data["width"], height=curr_data["height"])
        canvas_list = [gt_masks, pred_masks.squeeze()] 
        canvas = torch.cat(canvas_list, dim=1).squeeze(0).cpu().numpy()
        canvas = (canvas * 255).astype(np.uint8)
        imageio.imwrite(
            f"{train_path}/t_{timestep}_iter_{iteration}_cam{first_camera_indx}_masks.png",
            canvas,
        )


def initialize_optimizer(params, variables):
    lrs = {
        'means': 0.00016 * variables['scene_radius'],
        'rgbs': 0.0025,
        'masks': 0.0025,
        'quats': 0.001,
        'opacities': 0.05,
        'scales': 0.005,
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



def get_loss(params, curr_data, variables, is_initial_timestep, optimizer, strategy, strategy_state, i):
    losses = {}

    rendervar = params2rendervar(params)
    means = rendervar["means"]
    colors = rendervar["colors"]
    quats = rendervar["quats"]
    opacities = rendervar["opacities"]
    scales = rendervar["scales"]
    masks =params["masks"]
    cam_to_worlds = curr_data["c2w"]
    im, alphas, info = rasterize_splats(means, quats, scales, opacities, colors,
                                            camtoworlds=cam_to_worlds, Ks=curr_data["Ks"],
                                            width=curr_data["width"], height=curr_data["height"])
    if is_initial_timestep:
        pred_masks,_,_ = rasterize_splats(means, quats, scales, opacities, torch.sigmoid(masks.expand(-1, 3)),
                                                camtoworlds=cam_to_worlds, Ks=curr_data["Ks"],
                                                width=curr_data["width"], height=curr_data["height"])

        strategy.step_pre_backward(params=params, optimizers=optimizer, 
                                   state=strategy_state, step=i, info=info) #retains gradients for 2d means
    
    gt_image = curr_data["image"]
    if is_initial_timestep:
        losses['im'] = 0.8 * l1_loss_v1(im.squeeze(), gt_image.squeeze()) + 0.2 * (1.0 - calc_ssim(im.squeeze(), gt_image.squeeze()))
        losses["seg"] = 0.8 * l1_loss_v1(
            pred_masks.squeeze(), curr_data["mask"][..., None].expand(-1, -1, 3)
        ) + 0.2 * (
            1.0
            - calc_ssim(
                pred_masks.squeeze(), curr_data["mask"][..., None].expand(-1, -1, 3)
            )
        )
    else:
        losses['im'] = l1_loss_v1(im.squeeze(), gt_image.squeeze())

    #NOTE: turn these off 
    if not is_initial_timestep:
        is_fg = (params['masks'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means'][is_fg]
        fg_rot = rendervar['quats'][is_fg]

        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
        losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                              variables["neighbor_weight"])

        losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
                                            variables["neighbor_weight"])

        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

        losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()

        bg_pts = rendervar['means'][~is_fg]
        bg_rot = rendervar['quats'][~is_fg]
        losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

        losses['soft_col_cons'] = l1_loss_v2(params['rgbs'], variables["prev_col"])

    loss_weights = {'im': 1.0, 'seg': 0.1, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 20.0,
                    'soft_col_cons': 0.01}
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    return loss, variables, info


def initialize_per_timestep(params, variables, optimizer):
    pts = params['means']
    rot = torch.nn.functional.normalize(params['quats'])
    new_pts = pts + (pts - variables["prev_pts"])
    new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    is_fg = params['masks'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgbs'].detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()

    new_params = {'means': new_pts, 'quats': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['masks'][:, 0] > 0.5
    init_fg_pts = params['means'][is_fg]
    init_bg_pts = params['means'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['quats'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['quats']).detach()
    params_to_fix = ['opacities', 'scales']
    for param_name, opt_class in optimizer.items():
        if param_name in params_to_fix:
            opt_class.param_groups[0]["lr"] = 0.0
    return variables


def train_captured(data_dir , dates, learn_masks, apply_mask):
    """Training script for the rose scene, specifically tailored from my custom dataset."""
    scene_name = data_dir.split("/")[-1]
    now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    exp_path = f"../output/Dynamic3DGS/{scene_name}"
    os.makedirs(exp_path, exist_ok=True)
    # md = json.load(open(f"{data_dir}/transforms_train.json", 'r'))
    # params, variables, images, cam_to_worlds_dict, intrinsics = initialize_params_and_get_data(data_dir, md, every_t)
    train_indices = torch.arange(0, len(dates))/(len(dates)-1)
    from captured_data_utils import DynamicParser, Dynamic_Dataset, initialize_params_captured, InfiniteNeuralODEDataSampler, SingleTimeDataset
    parser = DynamicParser(
        data_dir=data_dir,
        factor=3,
        normalize=True,
        test_every=8,
        align_timesteps = False,
        dates =dates,
        use_dense =False,
        apply_mask=apply_mask,
    )

    #This loads all the data and stores it in a dict
    dynamic_cam_batch_size = 30
    trainset = Dynamic_Dataset(
        parser=parser,
        split="train",
        is_reverse=False,
        downsample_factor=1,
        include_zero=False,
        cam_batch_size = dynamic_cam_batch_size,
        time_normalize_factor=1,
        return_mask =True
    )
    
    testset = Dynamic_Dataset(
        parser=parser,
        split="test",
        is_reverse=False,
        downsample_factor=1,
        prepend_zero=True,
        downsample_eval=True,
        cam_batch_size = -1, #only using 1 test camera per timestep
        time_normalize_factor=1,
        return_mask = True
    )
    scene_scale = parser.scene_scale * 1.1  * 1 #global_scale equals 1
    print("Scene scale:", scene_scale)
    plot_intervals = [1, 599, 1999, 6999, 9999, 19999, 29999] 
    params,variables = initialize_params_captured(parser, scene_scale, learn_masks)
    strategy = DefaultStrategy(verbose=True)
    strategy.refine_stop_iter = 5000 #original code base only prunes before iteration 5000

    device="cuda"
    timesteps = list(range(0, len(dates))) #assuming all cameras have the same number of timesteps
    num_timesteps = len(timesteps)
    print(f"training our method on {num_timesteps} timesteps")
    print(f"the training times are {train_indices}")
    optimizer = initialize_optimizer(params, variables)

    # Do the strategy stuff here
    strategy.check_sanity(params, optimizer)
    strategy_state = strategy.initialize_state() #initializes running state
    output_params = [] #contain the gaussians params for each timestep.
    for t in timesteps: #training on timestep t
        is_initial_timestep = (t == 0)
        cam_batch_size = 1 if is_initial_timestep else dynamic_cam_batch_size
        if is_initial_timestep:
            single_timetrainset = SingleTimeDataset(trainset, 0) 
            trainloader = DataLoader(
                single_timetrainset,
                batch_size=1, #temporal batch size
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
            )
            trainloader_iter = iter(trainloader)
        if not is_initial_timestep:
            #Recreate a new dataloader every time we are at a new timestep
            lst_of_prog_indices = [t]
            print(f"reconstructing from {t-1} to {t}")
            prog_sampler = InfiniteNeuralODEDataSampler(
                trainset,
                lst_of_prog_indices,
                shuffle = True
            )
            # temp_batch_size = cfg.temp_batch_size if cfg.progressive_batch_size else len(lst_of_prog_indices) 
            trainloader = DataLoader(
                    trainset, 
                    batch_size=1, # temp batch size
                    shuffle=False,
                    num_workers=4,
                    persistent_workers=True,
                    pin_memory=True,
                    sampler=prog_sampler,
                    collate_fn=trainset.custom_collate_fn
                )
            trainloader_iter = iter(trainloader)
            params, variables = initialize_per_timestep(params, variables, optimizer)

        num_iter_per_timestep = 30_000 if is_initial_timestep else 2000 # initial_time is 10_000 and per time is 2000
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        all_camera_indices_t0 = list(trainset.timestep_images[0].keys())
        random_image = trainset.timestep_images[0][all_camera_indices_t0[0]]
        height, width = random_image.shape[0], random_image.shape[1]
        intrinsics = torch.from_numpy(trainset.timestep_intrinsics[0][1][None]).to("cuda").to(torch.float32).expand(cam_batch_size, -1,-1)
        curr_data = {"height": height, "width": width, "Ks": intrinsics}
        for i in range(num_iter_per_timestep):
            # curr_data = get_batch(dataset) #randomly selects a camera and rasterize.
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            if is_initial_timestep:
                camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
                pixels = data["image"].to(device) # [1, H, W, 3 or 4]
                mask = data["mask"].to(device)
                curr_data["c2w"] = camtoworlds
                curr_data["image"] = pixels.squeeze()
                curr_data["mask"] = mask.squeeze()
            else:
                c2w_batch = data[0].to(device) #if cam batch size is -1, will sample all cameras.
                gt_images_batch = data[1].to(device)
                curr_data["c2w"] = c2w_batch
                curr_data["image"] = gt_images_batch

            loss, variables, info = get_loss(params, curr_data, variables, is_initial_timestep, optimizer, strategy, strategy_state, i)
            loss.backward()
            if is_initial_timestep:
                strategy.step_post_backward(params, optimizer, strategy_state, i, info) #growing and pruning

            every_i = 10
            if i % every_i == 0:
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "num_gauss": params['means'].shape[0]
                })
                progress_bar.update(every_i)
            with torch.no_grad():
                # pred_image = torch.clamp(pred_image, 0, 1).squeeze()
                # gt_image = torch.clamp(gt_image, 0, 1).squeeze() 
                # psnr = calc_psnr(pred_image, gt_image)
                # progress_bar.set_postfix({f"train img {cam_ind} mask PSNR": f"{psnr:.{2}f}",
                #                         "num_gauss": means.shape[0]})
                # progress_bar.update(every_i)
                for opt in optimizer.values():
                    opt.step()
                    opt.zero_grad(set_to_none=True)

            if i in plot_intervals: 
                render_imgs_captured(testset, params, variables, exp_path, t, i)       
                # continue

        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep:
            camtoworlds_all = parser.camtoworlds[5:-5] #+- 10 cameras from left and right
            variables = initialize_post_first_timestep(params, variables, optimizer)
            traj_height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=traj_height
            )  # [N, 3, 4]
            camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
            )  # [N, 4, 4]

            camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
            # camtoworlds_all = generate_360_path(num_timesteps=40)
            video_dir = f"{exp_path}/static_360_videos"
            os.makedirs(video_dir, exist_ok=True)
            # writer = imageio.get_writer(f"{video_dir}/static_traj.mp4", fps=30, macro_block_size=1)
            frames = []
            rendervar = params2rendervar(params)
            means = rendervar["means"]
            rgbs = rendervar["colors"]
            quats = rendervar["quats"]
            opacities = rendervar["opacities"]
            scales = rendervar["scales"]
            for i in range(len(camtoworlds_all)):
                camtoworlds = camtoworlds_all[i : i + 1].to("cuda")
                Ks = curr_data["Ks"]
                renders, _, _ = rasterize_splats(
                    means=means,
                    quats = quats,
                    opacities=opacities,
                    scales=scales,
                    colors=rgbs,
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=curr_data["width"],
                    height=curr_data["height"],
                )  # [1, H, W, 4]
                colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
                colors = colors.squeeze(0).detach().cpu().numpy()
                colors = (colors * 255).astype(np.uint8)
                frames.append(colors)
            # writer.close()
            imageio.mimsave(f"{video_dir}/static_traj_final.mp4", 
                            frames,
                            fps=30)

    with open(f"{exp_path}/cfg.txt", "w") as f:
        f.write(f"data_dir: {data_dir}\n")      # Write data_dir with a newline
        
    save_params(output_params, exp_path)

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--data_dir", "-d", default="")
    parser.add_argument("--subsample_factor", "-s", type=int, default=1)
    parser.add_argument("--apply_mask", action="store_true")
    
    args = parser.parse_args()
    data_dir = args.data_dir
    learn_masks =True
    dates = sorted([f for f in os.listdir(data_dir) if f.startswith("timelapse")])[::-1] #NOTE: very important to invert the list, because we compute the point clouds of fully_grown
    if args.subsample_factor > 1:
        print(f"subsampling the timesteps with factor of  {args.subsample_factor}")
        dates = dates[::args.subsample_factor]
    train_captured(data_dir, dates, learn_masks, args.apply_mask)
    torch.cuda.empty_cache()


# if __name__ == "__main__":
#     exp_name = "exp1"
#     for sequence in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:
#         train(sequence, exp_name)
#         torch.cuda.empty_cache()
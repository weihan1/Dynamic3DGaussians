"""
Use this script to reproduce the dynamic 3DGS baseline on our custom plant dataset.
Notice for our dataset we do not
"""
import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
from torch import Tensor
from typing import Optional, Tuple, Dict

# Use our own rasterizer for fair comparison
# from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from gsplat import rasterization
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
from argparse import ArgumentParser
import imageio
from helpers import knn

"""
TODO: Since i'm training last timestep on 27 cameras and middle timesteps on 9, modify the code to allow for that
like allow for specific indexing of cameras
"""
def get_dataset(t, images, cam_to_worlds_dict, intrinsics, is_reverse=True, selected_cam_ids = ["r_1", "r_2", "r_3", "r_4", 
                                                                                                "r_6", "r_7", "r_8", "r_9", "r_10"]):
    """
    Retrieve all cameras for timestep t.
    If t != 0, only retrieve the cameras from selected_cam_ids. We want to simulate 27 cameras on last timestep and only 9 
    otherwise.
    """
    # for c in range(len(md['fn'][t])):
    #     w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
    #     cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
    #     fn = md['fn'][t][c]
    #     im = np.array(copy.deepcopy(Image.open(f"./data/{seq}/ims/{fn}")))
    #     im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
    #     seg = np.array(copy.deepcopy(Image.open(f"./data/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
    #     seg = torch.tensor(seg).float().cuda()
    #     seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
    #     dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})

    data = {
        "K": intrinsics,
        "camtoworld": {},
        "image": {},
        "image_id": {}
    }
    
    for camera_id, img_list in images.items():
        if t == 0 or camera_id in selected_cam_ids:
            data["camtoworld"][camera_id] = cam_to_worlds_dict[camera_id]
            data["image"][camera_id] = torch.from_numpy(img_list[t]).float()
            # data["image_id"][camera_id] = self.image_ids_dict[camera_id][timestep]
    
    return data


def get_batch(todo_dataset, dataset):
    """
    Copies per-time dataset onto todo_dataset and randomly selects a camera from that timestep
    """
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


def initialize_params_and_get_data(data_dir, md, num_gaussians=100_000):
    """
    Use this when training on custom plant dataset.
    It has "camera_angle_x" and "frames" as keys
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
        'means3D': points,
        'rgb_colors': rgbs,
        'unnorm_rotations':  quats, 
        'logit_opacities': opacities,
        'log_scales': scales,
    }

    params = {k: torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(), 
                 'scene_scale': scene_scale}

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
        Rasterize the splats, if appeareance optimization is on, add the color contribution from the appearance 
        module with the color of the splat.

        Return 
        -Rasterized image
        -Rendered opacity mask 
        -Additional rendering infos
        """
        #activate scales and opacities
        scales = torch.exp(scales)  # [N, 3]
        opacities = torch.sigmoid(opacities)  # [N,]
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
        'means3D': 0.00016 * variables["scene_scale"],
        'rgb_colors': 0.0025,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05, 
        'log_scales': 0.005 #changed from 0.001
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(params, curr_data, variables, is_initial_timestep):
    losses = {}

    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))

    if not is_initial_timestep:
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]

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

        losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()

        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])

    loss_weights = {'im': 1.0, 'seg': 3.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 20.0,
                    'soft_col_cons': 0.01}
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables


def initialize_per_timestep(params, variables, optimizer):
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    new_pts = pts + (pts - variables["prev_pts"])
    new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    #this prev offset calculates the distance between each point and its neighbor. 
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None] #distance between each point and all of its neighbors
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()

    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    #1. calculates the distances and weights for each gaussian and its neighbors.
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn) #excludes itself as neighbor
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist) #(N, num_knn)
    neighbor_dist = np.sqrt(neighbor_sq_dist) #(N, num_knn)

    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables


def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)


def train(data_dir, exp):
    """Training script for the rose scene, specifically tailored from my custom dataset."""
    #TODO: write rendering results here 
    scene_name = data_dir.split("/")[-1]
    if os.path.exists(f"./output/{exp}/{scene_name}"):
        print(f"Experiment '{exp}' for scene '{scene_name}' already exists. Exiting.")
        return
    md = json.load(open(f"{data_dir}/transforms_train.json", 'r'))
    num_timesteps = 150 

    params, variables, images, cam_to_worlds_dict, intrinsics = initialize_params_and_get_data(data_dir, md)
    optimizer = initialize_optimizer(params, variables)
    output_params = []
    for t in range(num_timesteps): #training on timestep t
        dataset = get_dataset(t, images, cam_to_worlds_dict, intrinsics) #getting all cameras for time t
        todo_dataset = []
        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = 10000 if is_initial_timestep else 2000
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            curr_data = get_batch(todo_dataset, dataset)
            loss, variables = get_loss(params, curr_data, variables, is_initial_timestep)
            loss.backward()
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                if is_initial_timestep:
                    params, variables = densify(params, variables, optimizer, i)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)
    save_params(output_params, seq, exp)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--data_dir", "-d", default="/projects/4DTimelapse/Dynamic3DGaussiansBaseline/plant_data/rose_mini")
    parser.add_argument("--name", "-n", type=str, default="exp1")
    args = parser.parse_args()
    data_dir = args.data_dir
    exp_name = args.name
    train(data_dir, exp_name)
    torch.cuda.empty_cache()

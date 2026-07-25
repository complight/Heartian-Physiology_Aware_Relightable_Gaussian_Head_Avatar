import os,logging,time,sys,uuid,shutil,copy
import torch,torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim,l2_loss,TVloss,spatial_gradient_loss

from gaussian_renderer import network_gui
import gaussian_renderer
from scene import Head_Scene,GaussianHeadModel
from utils.general_utils import safe_state,save_image_L
from utils.graphics_utils import normal_from_depth_image
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams,init_args,add_more_argument
from metrics import evaluate
from render import render_set,render_multi_views
from scene.data_loader import TrackedData
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError: 
    TENSORBOARD_FOUND = False

import cv2
import json
import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks
from rppg_toolbox import _detrend_torch
from POS_WANG import POS_WANG, POS_WANG_simplified

def training(all_args, testing_epochs, saving_epochs, checkpoint_epochs, checkpoint, debug_from):
    first_iter = 0
    scene_name=(all_args.source_path).split(os.path.sep)[-1]
    print(f"Model initialization and Data reading....")
    
    os.makedirs(os.path.join(all_args.model_path,"logs"),exist_ok=True)
    
    train_dataset=TrackedData(args.source_path,args,split='train',pre_load=True)
    test_dataset=TrackedData(args.source_path,args,split='test',pre_load=True)

    if all_args.fps==-1:
        if "UBFC-rPPG" in args.source_path:
            # video_file=os.path.join(all_args.source_path,f'{os.path.basename(all_args.source_path)}.avi')
            video_file=os.path.join(all_args.source_path,'vid.avi')
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"🔧 VIDEL FPS: {fps}")
            all_args.fps=fps
        elif "MMPD" in args.source_path or "PURE" in args.source_path:
            all_args.fps=30
    
    gaussians = GaussianHeadModel(all_args.sh_degree,all_args)

    # Load well-trained baseline for rPPG modulation
    if all_args.enable_heartbeat_albedo:
        rppg_model_path = all_args.model_path
        baseline_model_path = rppg_model_path.rsplit('_', 1)[0] + '_baseline'
        
        assert os.path.exists(baseline_model_path), f"Baseline path not found: {baseline_model_path}"
        
        all_args.model_path = baseline_model_path
        scene = Head_Scene(all_args, gaussians, dataset=train_dataset, load_epoch=args.baseline_load_epoch)
        all_args.model_path = rppg_model_path
        scene.model_path = rppg_model_path
    else:
        scene = Head_Scene(all_args, gaussians, dataset=train_dataset)
    
    if all_args.epochs==0:
        all_args.epochs=all_args.iterations//train_dataset.data_len
        saving_epochs.append(all_args.epochs)
    all_args.iterations=all_args.epochs*train_dataset.__len__()

    scene_name_safe = scene_name.replace('/', '_').replace('\\', '_').replace(':', '')
    log_file_path=os.path.join(all_args.model_path,"logs",f"({time.strftime('%Y-%m-%d_%H-%M-%S')})_Epoch({all_args.epochs})_({scene_name_safe}).log")
    
    logging.basicConfig(filename=log_file_path, 
                        level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Experiment Configuration: {all_args}")
    
    saving_epochs = list(range(all_args.save_interval, all_args.epochs, all_args.save_interval))
    saving_epochs.append(args.epochs)

    saving_iterations,testing_iterations,checkpoint_iterations=[i*train_dataset.data_len for i in saving_epochs],\
        [i*train_dataset.data_len for i in testing_epochs],[i*train_dataset.data_len for i in checkpoint_epochs]
    
    print(f"saving iterations:{saving_iterations}")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, all_args)
    device = gaussians.device
    
    bg_color= 1 if all_args.white_background else 0
    background = torch.tensor([bg_color]*3, dtype=torch.float32, device=device)
    tb_writer = prepare_output_and_logger(all_args)
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    render_temp_path=os.path.join(all_args.model_path,"train_temp_rendering")
    gt_temp_path=os.path.join(all_args.model_path,"train_temp_gt")
    if os.path.exists(render_temp_path):
        shutil.rmtree(render_temp_path)
    if os.path.exists(gt_temp_path):
        shutil.rmtree(gt_temp_path)
    os.makedirs(render_temp_path,exist_ok=True)
    os.makedirs(gt_temp_path,exist_ok=True)
    
    all_args.position_lr_max_steps=all_args.iterations-1000
    all_args.densify_until_iter=all_args.iterations-500

    # Freeze FLAME model parameters during albedo modulation training
    if all_args.enable_heartbeat_albedo:
        for attr in ['shape_dirs', 'expression_dirs', 'pose_dirs', 'lbs_weights', 'r_eyelid_dirs', 'l_eyelid_dirs', 'flame_vertexes_shaped', 'flame_J_regressor']:
            tensor = getattr(gaussians, attr)
            setattr(gaussians, attr, torch.nn.Parameter(tensor.detach().to(device), requires_grad=False))
        # speed up
        for attr in ['_xyz', '_scaling', '_rotation', '_opacity', 'flame_scale', '_roughness', '_reflectance', '_albedo']:
            getattr(gaussians, attr).requires_grad_(False)
        for p in gaussians.Envmap.parameters():
            p.requires_grad_(False)
        if gaussians.with_param_net_smirk:
            for p in gaussians.flame_params_net.parameters():
                p.requires_grad_(False)
    
    gaussians.training_setup(all_args)

    # Setup for albedo modulation
    if all_args.enable_heartbeat_albedo:
        for param_group in gaussians.optimizer.param_groups:
            param_group['lr'] = 0.0  # freeze all baseline parameters

        # extract rpg mean from images for POS
        image_dir = os.path.join(all_args.source_path, "image")
        rgb_means = []
        for t in range(len(train_dataset)+len(test_dataset)):
            img_path = os.path.join(image_dir, f"{t:05d}.png")
            img = cv2.imread(img_path)  # BGR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # [H, W, 3]
            
            mask = img_rgb.sum(axis=2) > 0.05  # [H, W] bool - mask out the background
            if mask.sum() > 0:
                rgb_means.append(img_rgb[mask].mean(axis=0))  # [3]
            else:
                rgb_means.append(np.array([0.0, 0.0, 0.0]))
        rgb_means = np.array(rgb_means)

        # Prepare skin Gaussian mask + load GT PPG signal + POS unsupervised estimation
        if "UBFC-rPPG" in args.source_path:
            skin_indices = gaussians.compute_skin_gaussian_indices(dist_threshold=0.006)

            gt_path = os.path.join(all_args.source_path, "ground_truth.txt")
            with open(gt_path, 'r') as f:
                lines = f.readlines()
            gaussians.gt_ppg_signal = np.array([float(x) for x in lines[0].split()])
            print(f"🔧 GT PPG loaded: {len(gaussians.gt_ppg_signal)} frames")

            gaussians.phase_pos, pos_signal = POS_WANG_simplified(rgb_means, all_args.fps)

        elif "MMPD" in args.source_path:
            skin_indices = gaussians.compute_skin_gaussian_indices(dist_threshold=0.016)
            gaussians.visualize_skin_gaussians(all_args.source_path, skin_indices)
            
            f = sio.loadmat(os.path.join(all_args.source_path, os.path.basename(all_args.source_path) + '.mat'), variable_names=['GT_ppg'])
            gaussians.gt_ppg_signal = f['GT_ppg'].reshape(-1)
            # skip data artifact
            skipped_path = os.path.join(all_args.source_path, 'skipped_indices.npy')
            if os.path.exists(skipped_path):
                skipped_indices = np.load(skipped_path)
                if len(skipped_indices) > 0:
                    print(f"🔧 Skip MMPD artifact data index: {skipped_indices}")
                    gaussians.gt_ppg_signal = np.delete(gaussians.gt_ppg_signal, skipped_indices)

            print(f"🔧 GT PPG loaded: {len(gaussians.gt_ppg_signal)} frames")

            gaussians.phase_pos, pos_signal = POS_WANG(all_args.fps, rgb_means)
        
        elif "PURE" in args.source_path:
            skin_indices = gaussians.compute_skin_gaussian_indices(dist_threshold=0.01)

            json_path = os.path.join(all_args.source_path, os.path.basename(all_args.source_path) + '.json')
            with open(json_path, 'r') as f:
                labels = json.load(f)
            waves = np.array([label["Value"]["waveform"] for label in labels["/FullPackage"]])

            target_length = len(train_dataset) + len(test_dataset)
            gaussians.gt_ppg_signal = np.interp(np.linspace(1, waves.shape[0], target_length), np.linspace(1, waves.shape[0], waves.shape[0]), waves)
            print(f"🔧 GT PPG loaded: {len(gaussians.gt_ppg_signal)} frames")

            gaussians.phase_pos, pos_signal = POS_WANG_simplified(rgb_means, all_args.fps)

        train_timesteps = train_dataset.split_indices
        train_timesteps_tensor = torch.tensor(train_timesteps, dtype=torch.long, device=device)
        gaussians.gt_window = torch.tensor([gaussians.gt_ppg_signal[t] for t in train_timesteps], dtype=torch.float32, device=device)

        gt_detrended = _detrend_torch(gaussians.gt_window, 100)
        gt_norm = (gt_detrended - gt_detrended.mean()) / (gt_detrended.std() + 1e-6)

        # signal_range = gaussians.gt_ppg_signal.max() - gaussians.gt_ppg_signal.min()
        # prominence=signal_range * 0.3
        
        peaks, _ = find_peaks(gaussians.gt_ppg_signal, distance=int(all_args.fps * 0.4))
        # linear interpolation
        theta_gt = np.zeros(len(gaussians.gt_ppg_signal))
        for i in range(len(peaks) - 1):
            p0, p1 = peaks[i], peaks[i+1]
            theta_gt[p0:p1] = np.linspace(0, 2*np.pi, p1 - p0, endpoint=False)
        theta_gt[:peaks[0]] = 0
        theta_gt[peaks[-1]:] = 2*np.pi

        gaussians.gt_phase = torch.tensor(theta_gt, dtype=torch.float32, device=device)

        frame_to_beat = np.zeros(len(gaussians.gt_ppg_signal), dtype=int)
        for i in range(len(peaks) - 1):
            frame_to_beat[peaks[i]:peaks[i+1]] = i
        frame_to_beat[:peaks[0]] = 0
        frame_to_beat[peaks[-1]:] = len(peaks) - 1

        gaussians.frame_to_beat = torch.tensor(frame_to_beat, dtype=torch.long, device=device)
        gaussians.num_beats = len(peaks)
        print(f"number of beats: {gaussians.num_beats}")

        # Align the POS estimation on original point
        pos_peaks, _ = find_peaks(pos_signal, distance=int(all_args.fps * 0.4))
        pos_phase_at_peaks = gaussians.phase_pos[pos_peaks]
        print(f"🔧 POS phase at GT peaks: mean={pos_phase_at_peaks.mean():.4f}, std={pos_phase_at_peaks.std():.4f}")
        print(f"🔧 offset = {np.degrees(pos_phase_at_peaks.mean()):.1f} degrees")

        offset = pos_phase_at_peaks.mean()
        gaussians.phase_pos = np.arctan2(np.sin(gaussians.phase_pos - offset), np.cos(gaussians.phase_pos - offset))

        gaussians.init_heartbeat_albedo(num_timesteps=len(train_dataset)+len(test_dataset), fps=all_args.fps, target_indices=skin_indices)
        gaussians.init_beat_embedding()
        gaussians.training_setup_heartbeat(all_args)
        gaussians.visualize_pos_extracted_phase(all_args.source_path)
        gaussians.visualize_skin_gaussians(all_args.source_path, skin_indices)

        gaussians.green_history = torch.zeros(len(train_timesteps) + len(test_dataset), device=device)
        print("🔧 Pre-filling green_history from baseline...")
        with torch.no_grad():
            for t in train_timesteps:
                modulated = gaussians.get_heartbeat_albedo(t, all_args.fps)
                gaussians.green_history[t] = modulated[:, 1].mean().detach()
        print(f"🔧 Pre-filled {len(gaussians.green_history)} frames")

    ema_imloss_for_log,ema_exploss_for_log = 0.0,0.0
    progress_bar = tqdm(range(first_iter, all_args.iterations), desc="Training progress",position=0)
    first_iter += 1
    epoch_loss,epoch_losses=0.0,[]
    epoch_phase_loss, epoch_pearson_loss, epoch_mse_loss, epoch_dev_loss = 0.0, 0.0, 0.0, 0.0

    logging.info(f"Start trainning....")
    gaussians.set_eval(False)
    render=gaussian_renderer.render_with_deferred
    train_stack = [item for item in train_dataset]

    for iteration in range(first_iter, all_args.iterations + 1): 
        epoch=iteration//train_dataset.data_len       
        iter_start.record()
        gaussians.update_learning_rate(iteration,all_args)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a data
        if not train_stack:
            train_stack = [item for item in train_dataset]
        viewpoint_cam_param = train_stack.pop(randint(0, len(train_stack)-1))
        
        gaussians.Envmap.update()
        # Render
        if (iteration - 1) == debug_from:
            all_args.debug = True
            
        gt_image = viewpoint_cam_param.original_image.cuda(device)
        gt_alpha_mask=viewpoint_cam_param.gt_alpha_mask.cuda(device)
        if all_args.random_background:
            bg = torch.rand((3), device=gaussians.device)
            gt_image = gt_image * (1 - gt_alpha_mask) + bg[:,None,None] * gt_alpha_mask
        else:
            bg = background

        render_pkg = render(viewpoint_cam_param, gaussians, all_args, bg, iteration=iteration)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        loss=0.0
        
        Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - all_args.lambda_dssim) * Ll1 + all_args.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss+=image_loss
        cam_o=viewpoint_cam_param.camera_center.cuda(device)
        
        
        if gaussians.with_param_net_smirk :
            jaw_pose_loss=all_args.lambda_jaw_pose*(gaussians.d_jaw_params**2).sum().mean()
            loss+=jaw_pose_loss

        if all_args.with_intrinsic_supervise and iteration>all_args.warm_up_iter:
            if viewpoint_cam_param.albedo is not None:
                persudo_albedo=viewpoint_cam_param.albedo.cuda(device)
                loss+=l1_loss(render_pkg["albedo"], persudo_albedo)*all_args.lambda_intrinsic_albedo

        if all_args.with_depth_supervise  :
            depth_render=render_pkg["depth"]
            normal_render=render_pkg["normal"]
            intrinsic_matrix, extrinsic_matrix = viewpoint_cam_param.intrinsic_matrix, viewpoint_cam_param.extrinsic_matrix
            normal_refer = normal_from_depth_image(depth_render[0], intrinsic_matrix.to(device), extrinsic_matrix.to(device)).permute(2,0,1)
            normal_refer=(normal_refer*(render_pkg["alpha"].detach()))
            if all_args.detach_normal_refer:
                normal_refer=normal_refer.detach()
            noraml_loss=((1-torch.sum(normal_refer*normal_render,dim=0))*(gt_alpha_mask>0.99)).mean()*all_args.lambda_normal
            loss+=noraml_loss
            
            
        if  iteration>all_args.warm_up_iter and all_args.with_envmap_consist:
            difmap,spemap=gaussians.Envmap.diffuse_map,gaussians.Envmap.specular_map
            spemap=torch.nn.functional.interpolate(spemap.permute(0,3,1,2),size=(difmap.shape[1], difmap.shape[2]), mode='bilinear',)
            envmap_consist_loss=l2_loss(difmap,spemap.permute(0,2,3,1))*all_args.lambda_envmap_consist
            loss+=envmap_consist_loss
            
        if all_args.with_tv_roughness and iteration>all_args.warm_up_iter:
            tv_normal_loss=TVloss(render_pkg["roughness"][None],gt_alpha_mask[None]*render_pkg["alpha"][None])*all_args.lambda_tv_roughness
            loss+=tv_normal_loss

        # PPG signals losses
        if all_args.enable_heartbeat_albedo and gaussians.gt_ppg_signal is not None:
            modulated_albedo = gaussians.get_heartbeat_albedo(viewpoint_cam_param.timestep, all_args.fps)  # [M, 3]
            skin_green = modulated_albedo[:, 1]  # green channel [M]
            gaussians.green_history[viewpoint_cam_param.timestep] = skin_green.mean().detach()

            # Phase loss
            pred_deltas = torch.nn.functional.softplus(gaussians.heartbeat_raw_deltas)
            pred_phase = gaussians.heartbeat_phase + torch.cumsum(pred_deltas, dim=0)
            pred_phase_train = pred_phase[train_timesteps_tensor]

            pred_theta_train = pred_phase_train % (2 * torch.pi)
            gt_phase_train = gaussians.gt_phase[train_timesteps_tensor]

            phase_diff = torch.atan2(torch.sin(pred_theta_train - gt_phase_train), torch.cos(pred_theta_train - gt_phase_train))
            phase_loss = (phase_diff ** 2).mean() * all_args.lambda_ppg_phase
            loss += phase_loss

            pred_theta_all = pred_phase % (2 * torch.pi)
            pos_phase_tensor = torch.tensor(gaussians.phase_pos, dtype=torch.float32, device=device)
            phase_diff_pos = torch.atan2(torch.sin(pred_theta_all - pos_phase_tensor), torch.cos(pred_theta_all - pos_phase_tensor))
            pos_weak_loss = (phase_diff_pos ** 2).mean() * all_args.lambda_pos_weak
            loss += pos_weak_loss

            # Waveform Pearson loss
            current_t = viewpoint_cam_param.timestep
            pred_window = gaussians.green_history[train_timesteps_tensor].clone()
            current_idx = train_timesteps.index(current_t)
            pred_window[current_idx] = skin_green.mean()

            pred_detrended = _detrend_torch(pred_window, 100)
            pred_norm = (pred_detrended - pred_detrended.mean()) / (pred_detrended.std() + 1e-6)

            pearson = (pred_norm * gt_norm).mean()
            pearson_loss = (1 - pearson) * all_args.lambda_ppg_pearson
            loss += pearson_loss

            mse_loss = torch.nn.functional.mse_loss(pred_norm, gt_norm) * all_args.lambda_gt_ppg
            loss += mse_loss

            # Fundamental waveform loss
            all_theta = torch.atan2(torch.sin(pred_phase_train), torch.cos(pred_phase_train)).detach()  # [T]

            sigma1 = torch.nn.functional.softplus(gaussians.heartbeat_log_sigma1)
            sigma2 = torch.nn.functional.softplus(gaussians.heartbeat_log_sigma2)

            d1 = torch.atan2(torch.sin(all_theta - gaussians.heartbeat_mu1), torch.cos(all_theta - gaussians.heartbeat_mu1))
            d2 = torch.atan2(torch.sin(all_theta - gaussians.heartbeat_mu2), torch.cos(all_theta - gaussians.heartbeat_mu2))

            A1 = torch.nn.functional.softplus(gaussians.heartbeat_A1)
            A2 = torch.nn.functional.softplus(gaussians.heartbeat_A2)

            g1 = A1 * torch.exp(-0.5 * (d1 / sigma1) ** 2)
            g2 = A2 * torch.exp(-0.5 * (d2 / sigma2) ** 2)
            fundamental_signal = g1 + g2  # [T]

            fundamental_detrended = _detrend_torch(fundamental_signal, 100)
            fund_norm = (fundamental_detrended - fundamental_detrended.mean()) / (fundamental_detrended.std() + 1e-6)

            pearson_f = (fund_norm * gt_norm).mean()
            pearson_loss_f = (1 - pearson_f) * all_args.lambda_ppg_pearson
            loss += pearson_loss_f

            mse_loss_f = torch.nn.functional.mse_loss(fund_norm, gt_norm) * all_args.lambda_gt_ppg
            loss += mse_loss_f

            # Regularization
            # 2 >= sig2 > sig1 >= 0
            sig1_sig2_reg = torch.clamp(sigma1 - sigma2, min=0) ** 2
            sigma_reg = torch.clamp(0.1 - sigma1, min=0) ** 2 + torch.clamp(sigma2 - 2.0, min=0) ** 2
            loss += (sigma_reg + sig1_sig2_reg) * all_args.lambda_sigma_reg

            # 1 >= A1 > A2 >= 0
            upA_reg = torch.clamp(A1 - 1.0, min=0) ** 2 + torch.clamp(A2 - 1.0, min=0) ** 2
            loss += upA_reg* all_args.lambda_A_reg
            
            if iteration > all_args.iterations / 2:
                a1_a2_reg = torch.clamp(A2 - A1, min=0) ** 2
                loss += a1_a2_reg * all_args.lambda_A_reg
            else:
                a1_a2_reg = torch.tensor(0.0)

            # mu2 > mu1
            mu1_mu2_reg = torch.clamp(gaussians.heartbeat_mu1 - gaussians.heartbeat_mu2, min=0) ** 2
            loss += mu1_mu2_reg * all_args.lambda_mu_reg

        loss.backward()

        iter_end.record()
        epoch_loss+=loss.item()
        if all_args.enable_heartbeat_albedo:
            epoch_phase_loss += phase_loss.item()
            epoch_pearson_loss += pearson_loss.item()
            epoch_mse_loss += mse_loss.item()

        if iteration % train_dataset.data_len == 0:
            tb_writer.add_scalar('train_epoch_losses', epoch_loss/train_dataset.data_len, epoch)
            logging.info(f"[Epoch {epoch}] loss: {epoch_loss/train_dataset.data_len}")
            logging.info(f"[Epoch {epoch}] Guassian points' number: {gaussians._xyz.shape[0]}")

            if all_args.enable_heartbeat_albedo and gaussians.gt_ppg_signal is not None:
                tb_writer.add_scalar('train_epoch_losses/phase_loss', epoch_phase_loss / train_dataset.data_len, epoch)
                tb_writer.add_scalar('train_epoch_losses/pearson_loss', epoch_pearson_loss / train_dataset.data_len, epoch)
                tb_writer.add_scalar('train_epoch_losses/ppg_mse_loss', epoch_mse_loss / train_dataset.data_len, epoch)

                logging.info(f"[Epoch {epoch}] phase_loss: {epoch_phase_loss/train_dataset.data_len:.6f}, pearson_loss: {epoch_pearson_loss/train_dataset.data_len:.6f}, ppg_mse: {epoch_mse_loss/train_dataset.data_len:.6f}, albedo_deviation_loss: {epoch_dev_loss/train_dataset.data_len:.6f}")
    
            epoch_loss=0.0
            epoch_phase_loss = 0.0
            epoch_pearson_loss = 0.0
            epoch_mse_loss = 0.0
        with torch.no_grad():
            if iteration%200==0 or iteration==1:
                torchvision.utils.save_image(image, os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name.replace('/', '_').replace('\\', '_') + ".png"))
                torchvision.utils.save_image(gt_image, os.path.join(gt_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name.replace('/', '_').replace('\\', '_') + ".png"))
                
                if iteration>all_args.warm_up_iter :
                    torchvision.utils.save_image((render_pkg["normal"]+1)/2, os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name.replace('/', '_').replace('\\', '_') + "_render_normal.png"))
                    if"albedo" in render_pkg.keys(): 
                        torchvision.utils.save_image((render_pkg["albedo"]), os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name.replace('/', '_').replace('\\', '_') + "_render_albedo.png"))

                if all_args.with_depth_supervise and iteration>all_args.warm_up_iter :
                    torchvision.utils.save_image((normal_refer+1)/2, os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name.replace('/', '_').replace('\\', '_') + "_refer_normal.png"))
                    depth_render=(depth_render-depth_render.min())/(depth_render.max()-depth_render.min())
                    save_image_L(depth_render,os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name.replace('/', '_').replace('\\', '_') + "_render_depth.png"))

                if  iteration>all_args.warm_up_iter:
                    save_image_L(render_pkg["roughness"],os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name.replace('/', '_').replace('\\', '_') + "_render_roughness.png"))
                    save_image_L(render_pkg["fresnel_reflectance"],os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam_param.image_name.replace('/', '_').replace('\\', '_') + "_render_reflectance.png"))
            # Progress bar
            ema_imloss_for_log = 0.4 * image_loss.item() + 0.6 * ema_imloss_for_log
            
            if iteration % 10 == 0:
                
                loss_dict = {
                    "Img_loss": f"{ema_imloss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                if all_args.enable_heartbeat_albedo:
                    loss_dict["phase_mse"] = f"{phase_loss:.{5}f}"
                    loss_dict["rPPG_mse"] = f"{mse_loss:.{5}f}"
                    loss_dict["rPPG_person"] = f"{pearson_loss:.{5}f}"
                    loss_dict["fun_mse"] = f"{mse_loss_f:.{5}f}"
                    loss_dict["fun_person"] = f"{pearson_loss_f:.{5}f}"
                    loss_dict["pos_weak_loss"] = f"{pos_weak_loss:.{5}f}"
                    loss_dict["A"] = f"{gaussians.heartbeat_A.item():.{5}f}"
                    loss_dict["A1"] = f"{A1.item():.{5}f}"
                    loss_dict["A2"] = f"{A2.item():.{5}f}"
                    loss_dict["sigma1"] = f"{sigma1.item():.{5}f}"
                    loss_dict["sigma2"] = f"{sigma2.item():.{5}f}"

                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == all_args.iterations:
                progress_bar.close()

            # Log and save
            gaussians.set_eval(True)
            training_report(tb_writer, iteration, epoch, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (all_args, background),train_dataset,test_dataset, image_loss,
                            phase_loss=phase_loss if all_args.enable_heartbeat_albedo else None,
                            pearson_loss=pearson_loss if all_args.enable_heartbeat_albedo else None,
                            mse_loss=mse_loss if all_args.enable_heartbeat_albedo else None,
                            pearson_loss_f=pearson_loss_f if all_args.enable_heartbeat_albedo else None,
                            mse_loss_f=mse_loss_f if all_args.enable_heartbeat_albedo else None,
                            sigma_reg=sigma_reg if all_args.enable_heartbeat_albedo else None,
                            sig1_sig2_reg=sig1_sig2_reg if all_args.enable_heartbeat_albedo else None,
                            mu1_mu2_reg=mu1_mu2_reg if all_args.enable_heartbeat_albedo else None,
                            a1_a2_reg=a1_a2_reg if all_args.enable_heartbeat_albedo else None,
                            upA_reg=upA_reg if all_args.enable_heartbeat_albedo else None,
                            pos_weak_loss=pos_weak_loss if all_args.enable_heartbeat_albedo else None,
                            gaussians=gaussians)
            gaussians.set_eval(False)
            
            if (iteration in saving_iterations):
                print("\n[EPOCH {} - ITER {}] Saving Gaussians".format(epoch,iteration))
                logging.info("\n[EPOCH {} - ITER {}] Saving Gaussians".format(epoch,iteration))
                scene.save(epoch)

            # Densification - only activate in baseline
            if iteration < all_args.densify_until_iter and not all_args.enable_heartbeat_albedo:
                # Keep track of max radii in image-space for pruning
                if gaussians.max_radii2D.shape[0]!=visibility_filter.shape[0]:
                    print(f"max_radii2D:{gaussians.max_radii2D.shape[0]},visibility_filter:{visibility_filter.shape[0]} \
                          ,xyz:{gaussians._xyz.shape[0]}")
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > all_args.densify_from_iter and iteration % all_args.densification_interval == 0:
                    size_threshold = 20 if iteration > all_args.opacity_reset_interval else None
                    gaussians.densify_and_prune(all_args.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % all_args.opacity_reset_interval == 0 or (all_args.white_background and iteration == all_args.densify_from_iter):
                    gaussians.reset_opacity()
                        

            # Optimizer step
            if iteration < all_args.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[EPOCH {} - ITER {}] Saving Checkpoint".format(epoch,iteration))
                logging.info("\n[EPOCH {} - ITER {}] Saving Checkpoint".format(epoch,iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(epoch) + ".pth")
    print("Training complete.")
    logging.info(f"Training complete.")

    # clear the memory for rendering and evaluation
    if hasattr(gaussians, 'geometry_cache'):
        del gaussians.geometry_cache
    
   
    if all_args.render_and_eval:
        gaussians.set_eval(True)
        gaussians.scene_name = scene_name_safe 

        with torch.no_grad():
            
            print("Rendering...")
            logging.info(f"Rendering training set....")
            render_set(all_args.model_path, "train", epoch, train_dataset, gaussians, all_args, bg,all_args, None)
            logging.info(f"Rendering testing set....")
            render_set(all_args.model_path, "test", epoch, test_dataset, gaussians, all_args, bg,all_args, None)
            
            # logging.info(f"Rendering training set with multiview....")
            # render_multi_views(all_args.model_path, "train", epoch, train_dataset, gaussians, all_args, bg)
            # logging.info(f"Rendering testing set with multiview....")
            # render_multi_views(all_args.model_path, "test", epoch, test_dataset, gaussians, all_args, bg)
            
            print("Evaluating on testing set....")
            logging.info(f"Evaluating on testing set....")
            evaluate([all_args.model_path])
            logging.info(f"Evaluating complete.")
        gaussians.set_eval(False)
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/events", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        os.makedirs(os.path.join(args.model_path, "envents"), exist_ok = True)
        tb_writer = SummaryWriter(os.path.join(args.model_path,"envents"))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration,epoch, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Head_Scene, renderFunc, renderArgs,train_dataset,test_dataset,image_loss, phase_loss=None, pearson_loss=None, mse_loss=None, pearson_loss_f=None, mse_loss_f=None, sigma_reg=None, sig1_sig2_reg=None, mu1_mu2_reg=None, a1_a2_reg=None, upA_reg=None, pos_weak_loss=None, gaussians=None): 
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

        tb_writer.add_scalar('train_loss_patches/image_loss', image_loss.item(), iteration)
        if phase_loss is not None:
            tb_writer.add_scalar('train_loss_rppg/phase_loss', phase_loss.item(), iteration)
        if pearson_loss is not None:
            tb_writer.add_scalar('train_loss_rppg/pearson_loss', pearson_loss.item(), iteration)
        if mse_loss is not None:
            tb_writer.add_scalar('train_loss_rppg/ppg_mse_loss', mse_loss.item(), iteration)
        if pearson_loss_f is not None:
            tb_writer.add_scalar('train_loss_rppg/pearson_loss_fun', pearson_loss_f.item(), iteration)
        if mse_loss_f is not None:
            tb_writer.add_scalar('train_loss_rppg/ppg_mse_loss_fun', mse_loss_f.item(), iteration)
        if sigma_reg is not None:
            tb_writer.add_scalar('train_loss_rppg/sigma_reg', sigma_reg.item(), iteration)
        if sig1_sig2_reg is not None:
            tb_writer.add_scalar('train_loss_rppg/sig1_sig2_reg', sig1_sig2_reg.item(), iteration)
        if mu1_mu2_reg is not None:
            tb_writer.add_scalar('train_loss_rppg/mu1_mu2_reg', mu1_mu2_reg.item(), iteration)
        if a1_a2_reg is not None:
            tb_writer.add_scalar('train_loss_rppg/a1_a2_reg', a1_a2_reg.item(), iteration)
        if upA_reg is not None:
            tb_writer.add_scalar('train_loss_rppg/upA_reg', upA_reg.item(), iteration)
        if pos_weak_loss is not None:
            tb_writer.add_scalar('train_loss_rppg/pos_weak_loss', pos_weak_loss.item(), iteration)
        
        if gaussians is not None and gaussians.enable_heartbeat_albedo:
            tb_writer.add_scalar('heartbeat_params/A', gaussians.heartbeat_A.item(), iteration)
            tb_writer.add_scalar('heartbeat_params/B', gaussians.heartbeat_B.item(), iteration)
            tb_writer.add_scalar('heartbeat_params/A1', torch.nn.functional.softplus(gaussians.heartbeat_A1).item(), iteration)
            tb_writer.add_scalar('heartbeat_params/A2', torch.nn.functional.softplus(gaussians.heartbeat_A2).item(), iteration)
            tb_writer.add_scalar('heartbeat_params/mu1', gaussians.heartbeat_mu1.item(), iteration)
            tb_writer.add_scalar('heartbeat_params/mu2', gaussians.heartbeat_mu2.item(), iteration)
            tb_writer.add_scalar('heartbeat_params/sigma1', torch.nn.functional.softplus(gaussians.heartbeat_log_sigma1).item(), iteration)
            tb_writer.add_scalar('heartbeat_params/sigma2', torch.nn.functional.softplus(gaussians.heartbeat_log_sigma2).item(), iteration)
            tb_writer.add_scalar('heartbeat_params/phase', gaussians.heartbeat_phase.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        logging.info(f"[EPOCH {epoch} - ITER {iteration}] Testing...")
        logging.info(f"[EPOCH {epoch} - ITER {iteration}] Guassian points' number:{scene.gaussians._xyz.shape[0]}")
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [test_dataset[idx]for idx in range(len(test_dataset))]}, 
                              {'name': 'train', 'cameras' : [train_dataset[idx % len(train_dataset)] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            logging.info(f"Start evaluate {config['name']} set...")
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(image.device), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[EPOCH {} - ITER {}] Evaluating {}: L1 {} PSNR {}".format(epoch,iteration, config['name'], l1_test, psnr_test))
                logging.info("\n[EPOCH {} - ITER {}] Evaluating {}: L1 {} PSNR {}".format(epoch,iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_epochs", nargs="+", type=int, default=[i*5 for i in range(1, 13)])
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[60])
    parser.add_argument("--save_interval", type=int, default=60)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser=add_more_argument(parser)
    args = parser.parse_args(sys.argv[1:])
    args.save_epochs.append(args.epochs)
    args.test_epochs.append(args.epochs)
    
    args=init_args(args)
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(args, args.test_epochs, args.save_epochs, args.checkpoint_epochs, args.start_checkpoint, args.debug_from,)

    # All done
    print("\nTraining complete.")
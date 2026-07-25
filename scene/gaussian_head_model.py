from scene.gaussian_model import *
from scene.flame import blend_shapes,batch_rodrigues,batch_rigid_transform
import os,pytorch3d.ops
import torch.nn.functional as F

from net_modules.flame_params_net_smirk import FlameParamsNetSmirk
from utils.sh_utils import RGB2SH
from net_modules.NVDIFFREC.envmap import EnvironmentMap
from utils.general_utils import get_minimum_axis, get_delayed_lr_func_with_decay, get_delayed_lr_func

from roma import rotmat_to_unitquat, quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
import time

import matplotlib
# Font configuration to match the conference requirement
import matplotlib.font_manager as fm
fm.fontManager.__init__()
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Linux Libertine' in available_fonts:
    matplotlib.rcParams['font.family'] = 'Linux Libertine'
else:
    print("Linux Libertine not found, using:", available_fonts[:5])
    matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.weight'] = 'normal'
matplotlib.rcParams['axes.titleweight'] = 'normal'
import matplotlib.pyplot as plt
from net_modules.heartbeat_mlp import HeartbeatMLP
from scipy.signal import hilbert
from rppg_toolbox import _process_signal

class GaussianHeadModel(GaussianModel):
    def __init__(self, sh_degree : int,args):
        super().__init__(sh_degree,args)
        self.device=args.device
        # self.preload=args.preload
        self.flame_vertexes=torch.empty(0)
        self.shape_dirs=torch.empty(0)
        self.expression_dirs=torch.empty(0)
        self.pose_dirs=torch.empty(0)
        self.lbs_weights=torch.empty(0)
        self.J_regressor=torch.empty(0)
        self.rot_parents=torch.empty(0)
        self.triangles=torch.empty(0)
        self.flame_scale=torch.empty(0)
        
        self.cache_xyz_canonical=None
        self.eval=False
        
        self.scaling_offset=None
        self.cached_shaped_vertexes=False
        self.lbs_return_transform_quad=False
        self.shaped_vertexes=None
        self.d_deform_scaling=None
        
        self.color_precomp=args.color_precomp
        self.with_param_net_smirk=args.with_param_net_smirk
        
        if self.with_param_net_smirk:
            self.flame_params_net=FlameParamsNetSmirk(**args.flame_params_net_params).to(self.device)
            
        self.with_depth_supervise=args.with_depth_supervise
        self.warm_up_iter=args.warm_up_iter
        
        self.Envmap=EnvironmentMap(args.diffuse_resolution,args.specular_resolution,args.mip_level,device=self.device)
        self.inverse_albedo_activation,self.inverse_roughness_activation = inverse_sigmoid,inverse_sigmoid
        self.albedo_activation,self.roughness_activation,self.reflectance_activation =torch.sigmoid, torch.sigmoid, torch.sigmoid
        self.max_reflectance,self.max_roughness=args.max_reflectance,args.max_roughness
        self.min_reflectance,self.min_roughness=args.min_reflectance,args.min_roughness

        self.enable_heartbeat_albedo = False
        self.current_timestep = None
        self.fps = args.fps
        self.phase_pos = None
        self.green_history = None
        self.gt_ppg_signal = None
        self.gt_phase = None
        self.frame_to_beat = None
        self.num_beats = None

    @property
    def get_deform_scaling(self):
        _scaling=self._scaling
        self.deform_scaling=self.scaling_activation(_scaling)*self.d_deform_scaling *self.flame_scale
        return self.deform_scaling
    
    @property
    def get_canonical_scaling(self):
        return self.get_scaling
        
    @property
    def get_deform_rotation(self):
        
        rot=self.rotation_activation(self._rotation)
        self.deform_rotation=self.rotation_activation(quat_xyzw_to_wxyz(quat_product(self.d_deform_rotation_xyzw,quat_wxyz_to_xyzw(rot))))
        return self.deform_rotation
    
    @property
    def get_deform_xyz(self):
        return self.deform_xyz*self.flame_scale
    

    @property
    def get_deform_opacity(self):
        return self.opacity_activation(self._opacity)+self.d_deform_opacity
    
    @property
    def get_minimum_axis(self):
        return get_minimum_axis(self.deform_scaling, self.deform_rotation)
    

    # @property
    # def get_albedo(self):
    #     return self.albedo_activation(self._albedo)

    # Modified: enable albedo modulation for embedding rPPG signals
    @property
    def get_albedo(self):
        if self.enable_heartbeat_albedo:
            modulated = self.get_heartbeat_albedo(self.current_timestep, self.fps)
            # write back after being activated to align correct values
            albedo = self.albedo_activation(self._albedo).clone()
            albedo[self.heartbeat_indices] = modulated
            return albedo
        return self.albedo_activation(self._albedo)
    
    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness)*(self.max_roughness-self.min_roughness)+self.min_roughness
    
    @property
    def get_reflectance(self):
        return self.reflectance_activation(self._reflectance)* \
            (self.max_reflectance-self.min_reflectance)+self.min_reflectance
    
    def get_deform_normal(self,cam_o):

        self.deform_normal=self.get_min_axis(cam_o,self.get_deform_rotation)
        return self.deform_normal
        
    def get_canonical_normal(self,cam_o):

        self._normal=self.get_min_axis(cam_o,self.get_rotation)
        return self._normal
 
    
    def create_from_pcd(self, head_scene_info, spatial_lr_scale,shape_params):
        super().create_from_pcd(head_scene_info.point_cloud, spatial_lr_scale)
        self.shape_param=shape_params.detach().clone()
        flame_mesh=head_scene_info.flame_mesh
        

        self.flame_scale=nn.Parameter(torch.tensor(head_scene_info.flame_scale,device=self.device,requires_grad=True))
        self.flame_vertexes=nn.Parameter(flame_mesh["v_template"].to(self.device).requires_grad_(False))
        self.shape_dirs=nn.Parameter(flame_mesh["shape_dirs"].to(self.device).requires_grad_(False))
        self.expression_dirs=nn.Parameter(flame_mesh["expression_dirs"].to(self.device).requires_grad_(False))
        self.pose_dirs=nn.Parameter(flame_mesh["pose_dirs"].to(self.device).requires_grad_(False))
        self.lbs_weights=nn.Parameter(flame_mesh["lbs_weights"].to(self.device).requires_grad_(False))
        self.J_regressor=nn.Parameter(flame_mesh["J_regressor"].T.to(self.device).requires_grad_(False))
        self.rot_parents=flame_mesh["parents"].to(self.device).long().requires_grad_(False)
        
        num_flame_vertexes=flame_mesh["v_template"].shape[0]
        self.triangles,self.flame_triangles=flame_mesh["triangles"],flame_mesh["flame_triangles"]
        
        r_eyelid_dirs,l_eyelid_dirs=torch.zeros_like(self.lbs_weights[:,:3]),torch.zeros_like(self.lbs_weights[:,:3])
        r_eyelid_dirs[:flame_mesh["r_eyelid"].shape[1],:]=flame_mesh["r_eyelid"]
        l_eyelid_dirs[:flame_mesh["l_eyelid"].shape[1],:]=flame_mesh["l_eyelid"]
        self.r_eyelid_dirs=nn.Parameter(r_eyelid_dirs.to(self.device).requires_grad_(False))
        self.l_eyelid_dirs=nn.Parameter(l_eyelid_dirs.to(self.device).requires_grad_(False))
        
        
        betas = shape_params
        self.flame_vertexes_shaped=(blend_shapes(betas,self.shape_dirs)+self.flame_vertexes).detach()
        
        self.flame_shape_dirs=self.shape_dirs.detach().clone()
        self.flame_expression_dirs=self.expression_dirs.detach().clone()
        self.flame_pose_dirs=self.pose_dirs.detach().clone()
        self.flame_lbs_weights=self.lbs_weights.detach().clone()
        self.flame_J_regressor=self.J_regressor.detach().clone()
        self.flame_r_eye_dirs=self.r_eyelid_dirs.detach().clone()
        self.flame_l_eye_dirs=self.l_eyelid_dirs.detach().clone()
        self.flame_vertex_normal=nn.Parameter(flame_mesh["v_normal"].to(self.device).requires_grad_(False))
        self.flame_vertex_idx_mask=flame_mesh["vertex_idx_mask"]
        shape_dirs=head_scene_info.point_cloud.shape_dirs.detach().clone().to(self.device)
        expression_dirs=head_scene_info.point_cloud.expression_dirs.detach().clone().to(self.device)
        pose_dirs=head_scene_info.point_cloud.pose_dirs.detach().clone().to(self.device)
        r_eyelid_dirs=head_scene_info.point_cloud.r_eyelid_dirs.detach().clone().to(self.device)
        l_eyelid_dirs=head_scene_info.point_cloud.l_eyelid_dirs.detach().clone().to(self.device)
        lbs_weights=head_scene_info.point_cloud.lbs_weights.detach().clone().to(self.device)
        self.flame_joint_center=torch.einsum('bik,ij->bjk', [self.flame_vertexes_shaped, self.flame_J_regressor])
        
        self.shape_dirs=nn.Parameter(shape_dirs,requires_grad=True)
        self.expression_dirs=nn.Parameter(expression_dirs,requires_grad=True)
        self.pose_dirs=nn.Parameter(pose_dirs,requires_grad=True)
        
        self.lbs_weights=nn.Parameter(lbs_weights,requires_grad=True)
        self.r_eyelid_dirs=nn.Parameter(r_eyelid_dirs,requires_grad=True)
        self.l_eyelid_dirs=nn.Parameter(l_eyelid_dirs,requires_grad=True)
        torch.cuda.empty_cache()
            

        self._albedo = nn.Parameter((torch.ones((self._xyz.shape[0], 3), device="cuda")*0.0).requires_grad_(True))
        _roughness=inverse_sigmoid(torch.ones((self._xyz.shape[0], 1), device="cuda")*0.9)
        _reflectance=inverse_sigmoid(torch.ones((self._xyz.shape[0], 1), device="cuda")*0.04)
        self._roughness = nn.Parameter(_roughness.requires_grad_(True))
        self._reflectance = nn.Parameter(_reflectance.requires_grad_( True))

    # Albedo modulation params for embedding PPG signals
    def init_heartbeat_albedo(self, num_timesteps, fps, target_indices):
        self.enable_heartbeat_albedo = True
        self.num_timesteps = num_timesteps

        self.heartbeat_indices = target_indices  # [M]

        # Learnable phase params
        self.heartbeat_phase = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.heartbeat_raw_deltas = nn.Parameter(
            torch.full((num_timesteps,), 2 * torch.pi * 1.5 / fps, device=self.device))
        
        # fundamental waveform params
        self.heartbeat_mu1 = nn.Parameter(torch.tensor(-0.5,  device=self.device))
        self.heartbeat_mu2 = nn.Parameter(torch.tensor(0.5, device=self.device))
        self.heartbeat_A1  = nn.Parameter(torch.tensor(0.2,  device=self.device)) # ~0.8 after softplus
        self.heartbeat_A2  = nn.Parameter(torch.tensor(-0.85,  device=self.device)) # ~0.35 after softplus
        self.heartbeat_log_sigma1 = nn.Parameter(torch.tensor(-2.0, device=self.device))  # ~0.12 after softplus
        self.heartbeat_log_sigma2 = nn.Parameter(torch.tensor(-0.5, device=self.device))  # ~0.47 after softplus
        self.heartbeat_A = nn.Parameter(torch.tensor(0.2, device=self.device))
        self.heartbeat_B = nn.Parameter(torch.tensor(0.5, device=self.device))

        # Spatial variance
        skin_xyz = self._xyz[self.heartbeat_indices].detach()
        xyz_min = skin_xyz.min(dim=0).values
        xyz_max = skin_xyz.max(dim=0).values
        self.skin_xyz_norm = (skin_xyz - xyz_min) / (xyz_max - xyz_min + 1e-6)
    
    def init_beat_embedding(self):
        # Residual MLP
        self.heartbeat_mlp = HeartbeatMLP(num_harmonics=10, hidden=128, num_beats=self.num_beats, latent_dim=8).to(self.device)

    def get_heartbeat_albedo(self, timestep, fps):

        if not self.enable_heartbeat_albedo or timestep is None:
            return None

        indices = self.heartbeat_indices
        # Base albedo (activated) for target Gaussians
        base_albedo = self.albedo_activation(self._albedo[indices])  # [M, 3]

        # Cumulative phase
        deltas = F.softplus(self.heartbeat_raw_deltas)
        cumulative_phase = self.heartbeat_phase + deltas[:timestep + 1].sum()

        # Unit cricle embedding
        x = torch.cos(cumulative_phase)
        y = torch.sin(cumulative_phase)
        theta = torch.atan2(y, x)

        d1 = torch.atan2(torch.sin(theta - self.heartbeat_mu1), torch.cos(theta - self.heartbeat_mu1))
        d2 = torch.atan2(torch.sin(theta - self.heartbeat_mu2), torch.cos(theta - self.heartbeat_mu2))

        sigma1 = F.softplus(self.heartbeat_log_sigma1)
        sigma2 = F.softplus(self.heartbeat_log_sigma2)

        A1 = torch.nn.functional.softplus(self.heartbeat_A1)
        A2 = torch.nn.functional.softplus(self.heartbeat_A2)

        g1 = A1 * torch.exp(-0.5 * (d1 / sigma1) ** 2)
        g2 = A2 * torch.exp(-0.5 * (d2 / sigma2) ** 2)
        fundamental = g1 + g2  # scalar

        phase_tensor = theta.reshape(1, 1)
        beat_idx = self.frame_to_beat[timestep]  # scalar
        beat_idx_tensor = torch.tensor([beat_idx], device=self.device)
        residual = self.heartbeat_mlp(phase_tensor, beat_idx_tensor, self.skin_xyz_norm).squeeze() 
        modulation = fundamental.detach() + self.heartbeat_B * residual  
        # modulation = fundamental

        # Apply the modulation factor
        modulated_albedo = base_albedo.clone()
        modulated_albedo[:, 1] = base_albedo[:, 1] + self.heartbeat_A * modulation

        # only in evaluation - for spatial variance visualization
        if hasattr(self, '_modulation_buffer'):
            signals = modulated_albedo[:, 1].clone()
            # signals = residual
            self._modulation_buffer[timestep] = signals.detach().cpu()

        return modulated_albedo 
            
    def training_setup(self, training_args):
        super().training_setup(training_args)
        self.optimizer.add_param_group({"params":[self.flame_scale],"lr":training_args.flame_scale_lr,"name":"flame_scale"})
       
        shape_dirs_params={"params":[self.shape_dirs],"lr":training_args.shape_dirs_lr,"name":"shape_dirs"}
        r_eyelid_dirs_params={"params":[self.r_eyelid_dirs],"lr":training_args.expression_dirs_lr,"name":"r_eyelid_dirs"}
        l_eyelid_dirs_params={"params":[self.l_eyelid_dirs],"lr":training_args.expression_dirs_lr,"name":"l_eyelid_dirs"}
        
        self.optimizer.add_param_group(shape_dirs_params)
        self.optimizer.add_param_group(r_eyelid_dirs_params)
        self.optimizer.add_param_group(l_eyelid_dirs_params)
        self.prune_params_names.append("shape_dirs")
        self.prune_params_names.append("l_eyelid_dirs")
        self.prune_params_names.append("r_eyelid_dirs")
        
        expression_dirs_params={"params":[self.expression_dirs],"lr":training_args.expression_dirs_lr,"name":"expression_dirs"}
        pose_dirs_params={"params":[self.pose_dirs],"lr":training_args.pose_dirs_lr,"name":"pose_dirs"}
        lbs_weights_params={"params":[self.lbs_weights],"lr":training_args.lbs_weights_lr,"name":"lbs_weights"}

        self.optimizer.add_param_group(expression_dirs_params)
        self.optimizer.add_param_group(pose_dirs_params)
        self.optimizer.add_param_group(lbs_weights_params)

        self.prune_params_names.append("expression_dirs")
        self.prune_params_names.append("pose_dirs")
        self.prune_params_names.append("lbs_weights")

        
        if self.with_param_net_smirk:
            flame_params_net_smirk_enocder_params={"params":self.flame_params_net.expression_encoder.encoder.parameters(),"lr":training_args.flame_params_net_smirk_encoder_lr,"name":"flame_params_net_smirk_encoder"}
            self.optimizer.add_param_group(flame_params_net_smirk_enocder_params)
            flame_params_net_smirk_decoder_params={"params":self.flame_params_net.expression_encoder.expression_layers.parameters(),"lr":training_args.flame_params_net_smirk_decoder_lr,"name":"flame_params_net_smirk_decoder"}
            self.optimizer.add_param_group(flame_params_net_smirk_decoder_params)
        
        self.shape_param=nn.Parameter(self.shape_param,requires_grad=False)
        
       
        
        self.optimizer.add_param_group({'params': list(self.Envmap.parameters()), 'lr': training_args.envmap_lr, "name": "Envmap"})
        
        self.optimizer.add_param_group({'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"})
        self.optimizer.add_param_group({'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"})
        self.optimizer.add_param_group({'params': [self._reflectance], 'lr': training_args.reflectance_lr, "name": "reflectance"})
        self.prune_params_names.extend(["roughness","albedo","reflectance"])
    
        self.albedo_schedule = get_expon_lr_func(lr_init=training_args.albedo_lr,
                                                lr_final=training_args.albedo_lr*training_args.appear_lr_decay,
                                                lr_delay_mult=training_args.position_lr_delay_mult,
                                                max_steps=training_args.position_lr_max_steps)
        self.roughness_schedule = get_expon_lr_func(lr_init=training_args.roughness_lr,
                                                lr_final=training_args.roughness_lr*training_args.appear_lr_decay,
                                                lr_delay_mult=training_args.position_lr_delay_mult,
                                                max_steps=training_args.position_lr_max_steps)
        self.reflectance_schedule = get_expon_lr_func(lr_init=training_args.reflectance_lr,
                                                lr_final=training_args.reflectance_lr*training_args.appear_lr_decay,
                                                lr_delay_mult=training_args.position_lr_delay_mult,
                                                max_steps=training_args.position_lr_max_steps)
        self.envmap_schedule = get_expon_lr_func(lr_init=training_args.envmap_lr,
                                                lr_final=training_args.envmap_lr*training_args.env_map_lr_decay,
                                                lr_delay_mult=training_args.position_lr_delay_mult,
                                                max_steps=training_args.position_lr_max_steps)
    
    # Add training setup for albedo modulation parameters
    def training_setup_heartbeat(self, training_args):
        self.optimizer.add_param_group({"params": [self.heartbeat_phase], "lr": training_args.heartbeat_phase_lr_3_init, "name": "heartbeat_phase"})
        self.optimizer.add_param_group({"params": [self.heartbeat_raw_deltas], "lr": training_args.heartbeat_mode3_raw_deltas_init, "name": "heartbeat_raw_deltas"})

        self.optimizer.add_param_group({"params": [self.heartbeat_mu1], "lr": training_args.heartbeat_mu1_lr_3_init, "name": "heartbeat_mu1"})
        self.optimizer.add_param_group({"params": [self.heartbeat_mu2], "lr": training_args.heartbeat_mu2_lr_3_init, "name": "heartbeat_mu2"})
        self.optimizer.add_param_group({"params": [self.heartbeat_A1], "lr": training_args.heartbeat_A1_lr_3_init, "name": "heartbeat_A1"})
        self.optimizer.add_param_group({"params": [self.heartbeat_A2], "lr": training_args.heartbeat_A2_lr_3_init, "name": "heartbeat_A2"})
        self.optimizer.add_param_group({"params": [self.heartbeat_log_sigma1], "lr": training_args.heartbeat_log_sig1_lr_3_init, "name": "heartbeat_log_sigma1"})
        self.optimizer.add_param_group({"params": [self.heartbeat_log_sigma2], "lr": training_args.heartbeat_log_sig2_lr_3_init, "name": "heartbeat_log_sigma2"})
        self.optimizer.add_param_group({"params": [self.heartbeat_B], "lr": training_args.heartbeat_B_lr_3, "name": "heartbeat_B"})
        self.optimizer.add_param_group({"params": [self.heartbeat_A], "lr": training_args.heartbeat_A_lr_3, "name": "heartbeat_A"})
        self.optimizer.add_param_group({"params": list(self.heartbeat_mlp.parameters()), "lr": training_args.heartbeat_mlp_lr_3_init, "name": "heartbeat_mlp"})

        # LR Schedulers
        self.heartbeat_raw_delta_scheduler = get_expon_lr_func(lr_init=training_args.heartbeat_mode3_raw_deltas_init,
                                                               lr_final=training_args.heartbeat_mode3_raw_deltas_final,
                                                               lr_delay_steps=0,
                                                               max_steps=training_args.heartbeat_fp_max_steps)
        self.heartbeat_phase_scheduler = get_expon_lr_func(lr_init=training_args.heartbeat_phase_lr_3_init,
                                                           lr_final=training_args.heartbeat_phase_lr_3_final,
                                                           lr_delay_steps=0,
                                                           max_steps=training_args.heartbeat_fp_max_steps)
        self.heartbeat_mlp_scheduler = get_expon_lr_func(lr_init=training_args.heartbeat_mlp_lr_3_init,
                                    lr_final=training_args.heartbeat_mlp_lr_3_final,
                                    lr_delay_steps=0,
                                    max_steps=training_args.heartbeat_mlp_max_steps)
        
    # Extract facial skin Gaussians utilizing FLAME union
    def compute_skin_vertex_indices(self):

        mask = self.flame_vertex_idx_mask

        face_vids = set(mask['face'].tolist())
        exclude_keys = ['left_eyeball', 'right_eyeball', 'left_eye_region', 'right_eye_region', 'left_ear', 'right_ear', 'scalp',]
        exclude_vids = set()
        for key in exclude_keys:
            if key in mask:
                exclude_vids.update(mask[key].tolist())

        skin_vids = sorted(face_vids - exclude_vids)
        return torch.tensor(skin_vids, dtype=torch.long, device=self.device)

    def compute_skin_gaussian_indices(self, dist_threshold=0.02):

        skin_vids = self.compute_skin_vertex_indices()  # [S]

        flame_verts = self.flame_vertexes_shaped
        if flame_verts.dim() == 3:
            flame_verts = flame_verts.squeeze(0)
        
        # skin vertices in canonical space
        skin_verts = flame_verts[skin_vids]  # [S, 3]
        gauss_xyz = self._xyz.detach()   # [N, 3]

        # KNN: match gaussians with the closest skin vertice
        dists, _, _ = pytorch3d.ops.knn_points(
            gauss_xyz.unsqueeze(0),    # [1, N, 3]
            skin_verts.unsqueeze(0),   # [1, S, 3]
            K=1
        )
        min_dists = dists[0, :, 0]  # [N]

        # ensure the closest vertex identified belongs to skin area
        all_verts = flame_verts  # [V, 3]
        dists_all, idx_all, _ = pytorch3d.ops.knn_points(
            gauss_xyz.unsqueeze(0),
            all_verts.unsqueeze(0),
            K=1
        )
        nearest_vid = idx_all[0, :, 0]  # [N]
        
        skin_vid_set = skin_vids.to(self.device)
        nearest_is_skin = torch.isin(nearest_vid, skin_vid_set)  # [N] bool

        front_face_mask = gauss_xyz[:, 2] > 0

        # ensure the gaussian close to the skin vertex in local and global comparision
        skin_mask = nearest_is_skin & (min_dists < dist_threshold ** 2) & front_face_mask
        
        skin_indices = torch.where(skin_mask)[0]
        print(f"Skin Gaussians: {len(skin_indices)} / {gauss_xyz.shape[0]} "
            f"({len(skin_indices)/gauss_xyz.shape[0]*100:.1f}%)")

        return skin_indices
    
    def visualize_skin_gaussians(self, source_path, skin_indices):
        
        gauss_xyz = self._xyz.detach()

        all_xyz = gauss_xyz.cpu().numpy()
        skin_xyz = gauss_xyz[skin_indices].cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        views = [('Front', 0, 1), ('Side', 2, 1), ('Top', 0, 2)]
        plt.subplots_adjust(wspace=0.02, left=0.02, right=0.98)
        
        for ax, (title, xi, yi) in zip(axes, views):
            ax.scatter(all_xyz[:, xi], all_xyz[:, yi], s=0.8, c='lightblue', alpha=0.3, label='All')
            ax.scatter(skin_xyz[:, xi], skin_xyz[:, yi], s=1.0, c='red', alpha=0.6, label='Skin')
            ax.set_title(title, fontsize=33, pad=10)
            ax.set_aspect('equal', adjustable='datalim')

            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(left=False, bottom=False)
        
        plt.tight_layout(pad=1.0)
        
        save_path = os.path.join(source_path, "skin_gaussians.png")
        plt.savefig(save_path, dpi=150)
        print("Saved skin Gaussian visualization image")
        plt.close()

    def visualize_pos_extracted_phase(self, source_path):

        t_axis = np.arange(len(self.gt_ppg_signal)) / self.fps

        fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

        # 1. GT PPG signal
        axes[0].plot(t_axis, self.gt_ppg_signal, color='orange', linewidth=0.8, label='GT PPG')
        axes[0].set_title('GT PPG Signal')
        axes[0].legend(); axes[0].grid(alpha=0.3)

        # 2. POS extracted signal
        # pos_signal = np.cos(self.phase_pos)
        axes[1].plot(t_axis, self.phase_pos, color='steelblue', linewidth=0.8, label='POS signal')
        axes[1].set_title('POS Extracted Signal Phase')
        axes[1].legend(); axes[1].grid(alpha=0.3)

        # 3. Phase comparison
        # Detrend + Bandpass filter
        gt_filtered = _process_signal(self.gt_ppg_signal, self.fps, use_bandpass = True)
        # Hilbert → instantaneous phase
        analytic = hilbert(gt_filtered)
        gt_phase = np.angle(analytic)

        axes[2].plot(t_axis, gt_phase, color='orange', linewidth=0.8, label='GT phase')
        axes[2].plot(t_axis, self.phase_pos, color='steelblue', linewidth=0.8, alpha=0.8, label='POS phase')
        axes[2].set_title('Phase Comparison')
        axes[2].legend(); axes[2].grid(alpha=0.3)
        axes[2].set_xlabel('Time (s)')

        plt.suptitle('POS vs GT Phase Diagnostic')
        plt.tight_layout()
        plt.savefig(os.path.join(source_path, "pos_phase_diagnostic-0.75.png"), dpi=150)
        plt.close()
        print("📊 POS phase diagnostic saved")

    def visualize_spatial_weights(self, output_path=None):
        if not self.enable_heartbeat_albedo or not hasattr(self, '_modulation_buffer'):
            print("No spatial delta available")
            return

        skin_xyz = self._xyz[self.heartbeat_indices].detach().cpu().numpy()

        all_mods = torch.stack(list(self._modulation_buffer.values()), dim=0)  # [T, M]
        modulation_map = all_mods.mean(dim=0).numpy()  # [M]
        vmin, vmax = modulation_map.min(), modulation_map.max()

        plt.figure(figsize=(5, 6))
        sc = plt.scatter(skin_xyz[:, 0], skin_xyz[:, 1], c=modulation_map, cmap='RdYlGn_r', vmin=vmin, vmax=vmax, s=2.0, alpha=0.8)
        cbar = plt.colorbar(sc, shrink=0.8)
        cbar.ax.tick_params(labelsize=20)
        plt.title(f'Spatial Intensity', fontsize=23.5)
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Spatial Intensity visualization saved")
        plt.close()
    
    def lbs_v2(self,xyz,shape_params=None, expression_params=None,full_pose_params=None,
            eyelid_params=None,translation_param=None,lbs_weight_t=None,):
        #Performs Linear Blend Skinning with the given shape and pose parameters        
        batch_size = shape_params.shape[0]
        device=self.device
        transform_dict={}
        #Full_ Pose [(global)3, neck (0)3, (jaw)3, eyepose (0)6]
        full_pose=full_pose_params
        expression_dirs,pose_dirs=self.expression_dirs,self.pose_dirs
        
        if self.eval:
            if self.cache_xyz_canonical is not None:
                xyz_canonical=self.cache_xyz_canonical
            else:
                xyz_canonical=xyz.unsqueeze(0).expand(batch_size, -1, -1)
                xyz_canonical=xyz_canonical+blend_shapes(shape_params,self.shape_dirs)
                self.cache_xyz_canonical=xyz_canonical.detach().clone()
            
        else :
            xyz_canonical=xyz.unsqueeze(0).expand(batch_size, -1, -1)
            xyz_canonical=xyz_canonical+blend_shapes(shape_params,self.shape_dirs)
        
        shape_dirs_t=expression_dirs#torch.cat([self.shape_dirs,expression_dirs],dim=2)
        betas = expression_params
        
        if lbs_weight_t is None:
            lbs_weight_t=self.lbs_weights
        pose_dirs_t=pose_dirs
        r_eyelid_dirs_t,l_eyeild_dirs_t=self.r_eyelid_dirs,self.l_eyelid_dirs
            
        shape_offsets=blend_shapes(betas,shape_dirs_t) # n 3 300+100
        xyz_shaped = xyz_canonical + shape_offsets

        # Get the joints 5 n
        J=self.flame_joint_center
        ident = torch.eye(3, dtype=torch.float32, device=device)
        rot_mats = batch_rodrigues(full_pose.view(-1, 3), dtype=torch.float32).view([batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])# 1 36
        # (b x l) x (m, 3 ,l) -> b x m x 3
        
        pose_offsets =blend_shapes(pose_feature,pose_dirs_t)
        xyz_posed = pose_offsets + xyz_shaped
        if eyelid_params is not None:
            xyz_posed = xyz_posed + r_eyelid_dirs_t.unsqueeze(0).expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
            xyz_posed = xyz_posed + l_eyeild_dirs_t.unsqueeze(0).expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]
        
        J_transformed, A = batch_rigid_transform(rot_mats, J, self.rot_parents, dtype=torch.float32)
        #add identity matrix
        A=torch.cat([A,torch.eye(4,device=device)[None,None,...].expand(batch_size,-1,-1,-1)],dim=1)
        W = lbs_weight_t.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        #add identity rotation joint
        num_joints = self.J_regressor.shape[1]+1
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
        homogen_coord = torch.ones([batch_size, xyz_posed.shape[1], 1],
                                dtype=torch.float32, device=device)
        v_posed_homo = torch.cat([xyz_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        xyz_lbs = v_homo[:, :, :3, 0]
        if translation_param is not None:
            xyz_lbs=xyz_lbs+translation_param
        transform_dict.update({"transform_matrix":T,"shape_offsets":shape_offsets,"pose_offsets":pose_offsets,"xyz_posed":xyz_posed,"canonical":xyz_canonical})
        
        if self.lbs_return_transform_quad:
            A3=A[0,:,:3,:3]
            A3_quad=rotmat_to_unitquat(A3)
            T_quad=torch.matmul(W, A3_quad.view(batch_size, num_joints, 4)).view(batch_size, -1, 4)
            transform_dict["transform_matrix_quad"]=T_quad
        if not self.eval :#and self.with_normal_attribute:
            expr_offsets=blend_shapes(expression_params,self.flame_expression_dirs)
            pose_offsets=blend_shapes(pose_feature,self.flame_pose_dirs)
            eyelid_offset=0.0
            if eyelid_params is not None:
                eyelid_offset=self.flame_l_eye_dirs.unsqueeze(0).expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]\
                    +self.flame_r_eye_dirs.unsqueeze(0).expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
            flame_vertex_posed=self.flame_vertexes_shaped.expand(batch_size, -1, -1)+expr_offsets+pose_offsets+eyelid_offset
            flame_W = self.flame_lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
            flame_T = torch.matmul(flame_W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
            flame_homogen_coord = torch.ones([batch_size, flame_vertex_posed.shape[1], 1],dtype=torch.float32, device=device)
            flame_posed_homo = torch.cat([flame_vertex_posed, flame_homogen_coord], dim=2)
            flame_v_homo = torch.matmul(flame_T, torch.unsqueeze(flame_posed_homo, dim=-1))
            self.flame_vertex_lbs = flame_v_homo[:, :, :3, 0].squeeze(0)
            
        return xyz_lbs,transform_dict

    
    
    def forward(self,shape_param,expression_param,full_pose_param,camera_center,eyelid_param=None,translation_param=None,warped_image=None,iteration=torch.inf):
        
        if self.shape_param is not None:
            shape_param=self.shape_param
        if self.with_param_net_smirk and warped_image is not None:
            start_time=time.time()
            out_params=self.flame_params_net(warped_image)
            end_time=time.time()
            self.params_extract_time=(end_time-start_time)
            expression_param = out_params['expression_params']
            jaw_params = out_params.get('jaw_params', None)
            eyelid_param = out_params.get('eyelid_params', None)
            #[(global)3, neck (0)3, (jaw)3, eyepose (0)6]
            #use_smirk_jaw_pose
            self.d_jaw_params=(full_pose_param[:,6:9]-jaw_params)
            full_pose_param[:,6:9]=jaw_params
            
        _xyz_t=self._xyz
           
        lbs_weights_exp=torch.relu(self.lbs_weights)
        self.lbs_weights_t=lbs_weights_exp / (torch.sum(lbs_weights_exp, dim=-1, keepdim=True)+1e-5)#F.softmax(self.lbs_weights,dim=1)

        _xyz_lbs,transform_dict=self.lbs_v2(_xyz_t,shape_param,expression_param,full_pose_param,
                                            eyelid_param,translation_param,lbs_weight_t=self.lbs_weights_t)
        self.deform_xyz=_xyz_lbs.squeeze(0)
        self.d_deform_rotation_xyzw=rotmat_to_unitquat(transform_dict["transform_matrix"][0,:,:3,:3])
        self.d_deform_scaling=1.0
        self.transform_dict=transform_dict

        self.d_deform_opacity=torch.tensor(0.,device=self.device)


            
    def save_model(self, path):
        
        super().save_ply(os.path.join(path,"point_cloud.ply"))
        atrributes_params_dict={}
        
        atrributes_params_dict["flame_scale"]=self.flame_scale
        atrributes_params_dict["flame_vertexes"]=self.flame_vertexes
        atrributes_params_dict["shape_dirs"]=self.shape_dirs
        atrributes_params_dict["expression_dirs"]=self.expression_dirs
        atrributes_params_dict["pose_dirs"]=self.pose_dirs
        atrributes_params_dict["lbs_weights"]=self.lbs_weights
        atrributes_params_dict["J_regressor"]=self.J_regressor
        atrributes_params_dict["rot_parents"]=self.rot_parents
        atrributes_params_dict["shape_param"]=self.shape_param
        atrributes_params_dict["flame_joint_center"]=self.flame_joint_center
        
        
        atrributes_params_dict["shape_dirs"]=self.shape_dirs
        atrributes_params_dict["expression_dirs"]=self.expression_dirs
        atrributes_params_dict["pose_dirs"]=self.pose_dirs
        atrributes_params_dict["lbs_weights"]=self.lbs_weights
        atrributes_params_dict["r_eyelid_dirs"]=self.r_eyelid_dirs
        atrributes_params_dict["l_eyelid_dirs"]=self.l_eyelid_dirs
        atrributes_params_dict["flame_J_regressor"]=self.flame_J_regressor
        atrributes_params_dict["flame_vertexes_shaped"]=self.flame_vertexes_shaped
            
        if self.with_param_net_smirk:
            torch.save(self.flame_params_net.state_dict(), os.path.join(path,"flame_params_net.pth"))
       

        atrributes_params_dict["albedo"]=self._albedo
        atrributes_params_dict["roughness"]=self._roughness
        atrributes_params_dict["reflectance"]=self._reflectance

        atrributes_params_dict["Envmap-diffuse_map"],atrributes_params_dict["Envmap-diffuse"]=self.Envmap.diffuse_map,self.Envmap.diffuse
        atrributes_params_dict["Envmap-specular_map"],atrributes_params_dict["Envmap-specular"]=self.Envmap.specular_map,self.Envmap.specular
        atrributes_params_dict["max_reflectance"],atrributes_params_dict["min_reflectance"]=self.max_reflectance,self.min_reflectance
        atrributes_params_dict["max_roughness"],atrributes_params_dict["min_roughness"]=self.max_roughness,self.min_roughness

        # Save heartbeat albedo parameters
        atrributes_params_dict["fps"] = torch.tensor(self.fps)
        
        if hasattr(self, 'enable_heartbeat_albedo') and self.enable_heartbeat_albedo:
            atrributes_params_dict["heartbeat_phase"] = self.heartbeat_phase
            atrributes_params_dict["heartbeat_raw_deltas"] = self.heartbeat_raw_deltas
            atrributes_params_dict["heartbeat_mu1"] = self.heartbeat_mu1
            atrributes_params_dict["heartbeat_mu2"] = self.heartbeat_mu2
            atrributes_params_dict["heartbeat_A1"] = self.heartbeat_A1
            atrributes_params_dict["heartbeat_A2"] = self.heartbeat_A2
            atrributes_params_dict["heartbeat_log_sigma1"] = self.heartbeat_log_sigma1
            atrributes_params_dict["heartbeat_log_sigma2"] = self.heartbeat_log_sigma2
            atrributes_params_dict["heartbeat_A"] = self.heartbeat_A
            atrributes_params_dict["heartbeat_B"] = self.heartbeat_B
            atrributes_params_dict["heartbeat_indices"] = self.heartbeat_indices
            atrributes_params_dict["heartbeat_num_timesteps"] = torch.tensor(self.num_timesteps)
            atrributes_params_dict["skin_xyz_norm"] = self.skin_xyz_norm

            atrributes_params_dict["num_beats"] = torch.tensor(self.num_beats)
            atrributes_params_dict["frame_to_beat"] = self.frame_to_beat
            torch.save(self.heartbeat_mlp.state_dict(), os.path.join(path, "heartbeat_mlp.pth"))

        torch.save(atrributes_params_dict, os.path.join(path,"attributes_params.pth"))
            
    def load_model(self, path):
        super().load_ply(os.path.join(path,"point_cloud.ply"))
        atrributes_params_dict=torch.load(os.path.join(path,"attributes_params.pth"),map_location="cpu")
        
        self.flame_scale=nn.Parameter(atrributes_params_dict["flame_scale"].to(self.device),requires_grad=True)
        self.flame_vertexes=nn.Parameter(atrributes_params_dict["flame_vertexes"].to(self.device),requires_grad=False)
        self.shape_dirs=nn.Parameter(atrributes_params_dict["shape_dirs"].to(self.device),requires_grad=False)
        self.expression_dirs=nn.Parameter(atrributes_params_dict["expression_dirs"].to(self.device),requires_grad=False)
        self.pose_dirs=nn.Parameter(atrributes_params_dict["pose_dirs"].to(self.device),requires_grad=False)
        self.lbs_weights=nn.Parameter(atrributes_params_dict["lbs_weights"].to(self.device),requires_grad=False)
        self.J_regressor=nn.Parameter(atrributes_params_dict["J_regressor"].to(self.device),requires_grad=False)
        self.rot_parents=atrributes_params_dict["rot_parents"].to(self.device).long()
        if "flame_joint_center" in atrributes_params_dict.keys():
            self.flame_joint_center=nn.Parameter(atrributes_params_dict["flame_joint_center"].to(self.device),requires_grad=False)
        if "shape_param" in atrributes_params_dict.keys():
            self.shape_param=nn.Parameter(atrributes_params_dict["shape_param"].to(self.device),requires_grad=False)
            
        
        self.shape_dirs=atrributes_params_dict["shape_dirs"].to(self.device)
        self.expression_dirs=atrributes_params_dict["expression_dirs"].to(self.device)
        self.pose_dirs=atrributes_params_dict["pose_dirs"].to(self.device)
        self.lbs_weights=atrributes_params_dict["lbs_weights"].to(self.device)
    
        self.r_eyelid_dirs=atrributes_params_dict["r_eyelid_dirs"].to(self.device)
        self.l_eyelid_dirs=atrributes_params_dict["l_eyelid_dirs"].to(self.device)
        self.flame_J_regressor=atrributes_params_dict["flame_J_regressor"]
        self.flame_vertexes_shaped=atrributes_params_dict["flame_vertexes_shaped"]

        
        if  self.with_param_net_smirk:
            statedict=torch.load( os.path.join(path,"flame_params_net.pth"),map_location="cpu")
            self.flame_params_net.load_state_dict(statedict)
            self.flame_params_net.to(self.device)
        

        self._albedo=nn.Parameter(atrributes_params_dict["albedo"].to(self.device),requires_grad=True)
        self._roughness=nn.Parameter(atrributes_params_dict["roughness"].to(self.device),requires_grad=True)
        self._reflectance=nn.Parameter(atrributes_params_dict["reflectance"].to(self.device),requires_grad=True)

        self.Envmap.diffuse_map=nn.Parameter(atrributes_params_dict["Envmap-diffuse_map"].to(self.device),requires_grad=True)
        self.Envmap.diffuse=atrributes_params_dict["Envmap-diffuse"].clone().detach().to(self.device)
        self.Envmap.specular_map=nn.Parameter(atrributes_params_dict["Envmap-specular_map"].to(self.device),requires_grad=True)
        self.Envmap.specular=atrributes_params_dict["Envmap-specular"].clone().detach().to(self.device)
        self.max_reflectance,self.min_reflectance=atrributes_params_dict["max_reflectance"],atrributes_params_dict["min_reflectance"]
        self.max_roughness,self.min_roughness=atrributes_params_dict["max_roughness"],atrributes_params_dict["min_roughness"]
        
        # Load albedo modulation parameters and fps
        self.fps = float(atrributes_params_dict["fps"].item())
        
        if "heartbeat_indices" in atrributes_params_dict:
            self.enable_heartbeat_albedo = True
            self.num_timesteps = int(atrributes_params_dict["heartbeat_num_timesteps"].item())
            self.heartbeat_indices = atrributes_params_dict["heartbeat_indices"].to(self.device)

            self.heartbeat_phase = nn.Parameter(atrributes_params_dict["heartbeat_phase"].to(self.device), requires_grad=True)
            self.heartbeat_raw_deltas = nn.Parameter(atrributes_params_dict["heartbeat_raw_deltas"].to(self.device), requires_grad=True)
            self.heartbeat_mu1 = nn.Parameter(atrributes_params_dict["heartbeat_mu1"].to(self.device), requires_grad=True)
            self.heartbeat_mu2 = nn.Parameter(atrributes_params_dict["heartbeat_mu2"].to(self.device), requires_grad=True)
            self.heartbeat_A1 = nn.Parameter(atrributes_params_dict["heartbeat_A1"].to(self.device), requires_grad=True)
            self.heartbeat_A2 = nn.Parameter(atrributes_params_dict["heartbeat_A2"].to(self.device), requires_grad=True)
            self.heartbeat_log_sigma1 = nn.Parameter(atrributes_params_dict["heartbeat_log_sigma1"].to(self.device), requires_grad=True)
            self.heartbeat_log_sigma2 = nn.Parameter(atrributes_params_dict["heartbeat_log_sigma2"].to(self.device), requires_grad=True)
            self.heartbeat_A = nn.Parameter(atrributes_params_dict["heartbeat_A"].to(self.device), requires_grad=True)
            self.heartbeat_B = nn.Parameter(atrributes_params_dict["heartbeat_B"].to(self.device), requires_grad=True)

            self.skin_xyz_norm = atrributes_params_dict["skin_xyz_norm"].to(self.device)

            self.num_beats = int(atrributes_params_dict["num_beats"].item())
            self.frame_to_beat = atrributes_params_dict["frame_to_beat"].clone().detach().to(self.device)

            self.heartbeat_mlp = HeartbeatMLP(num_harmonics=10, hidden=128, num_beats=self.num_beats, latent_dim=8).to(self.device)
            mlp_path = os.path.join(path, "heartbeat_mlp.pth")
            if os.path.exists(mlp_path):
                self.heartbeat_mlp.load_state_dict(torch.load(mlp_path, map_location="cpu"))
                self.heartbeat_mlp.to(self.device)

    def get_canonical_xyz(self):
        #save canonical xyz
        canonical_xyz=self._xyz
        return canonical_xyz
    
    def save_canonical_ply(self,path="./canonical_xyz.ply"):

        canonical_xyz=self.get_canonical_xyz()
        
        normals = np.zeros_like(canonical_xyz.detach().detach().cpu().numpy())
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc =  RGB2SH(self.albedo_activation(self._albedo)).detach().contiguous().cpu().numpy()
        
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        canonical_xyz=canonical_xyz.detach().cpu().numpy()
        elements = np.empty(canonical_xyz.shape[0], dtype=dtype_full)
        if f_rest.shape[0]!=canonical_xyz.shape[0]:
            f_rest=np.zeros((canonical_xyz.shape[0],f_rest.shape[1]))
        attributes = np.concatenate((canonical_xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        

    def cached_shaped_vertex(self,shape_param):
        vertexes=self.flame_vertexes
        shape_dirs_t=self.shape_dirs
        betas=shape_param
        self.shaped_vertexes=(blend_shapes(betas,shape_dirs_t).squeeze()+vertexes).detach()
        self.cached_shaped_vertexes=True
    
    def set_eval(self,state):
        self.eval=state
        if not state:
            self.cache_xyz_canonical=None

    def extra_loss(self):
        loss_dict={"d_opacity":0,"d_rotation":0,"d_scaling":0,"d_xyz":0}
        return loss_dict
    
    def get_min_axis(self, cam_o,rotation):
        pts = self.get_xyz
        p2o = cam_o[None] - pts
        scales = self.deform_scaling
        min_axis_id = torch.argmin(scales, dim = -1, keepdim=True)
        min_axis = torch.zeros_like(scales).scatter(1, min_axis_id, 1)
        rot_matrix = build_rotation(rotation)
        ndir = torch.bmm(rot_matrix, min_axis.unsqueeze(-1)).squeeze(-1)
        neg_msk = torch.sum(p2o*ndir, dim=-1) < 0
        ndir[neg_msk] = -ndir[neg_msk] # make sure normal orient to camera
        #ndir=get_minimum_axis(self.deform_scaling,rotation)

        return ndir
    
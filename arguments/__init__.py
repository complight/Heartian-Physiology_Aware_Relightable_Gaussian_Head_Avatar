"""
Copyright (c) 2025 Eastbean Zhang. All rights reserved.

This code is a modified version of the original work from:
https://github.com/graphdeco-inria/gaussian-splatting

"""

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 0
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.device = "cuda:0"
        self.eval = True
        self.color_precomp=False
        
        self.n_shape, self.n_expr=100,50 # 300 100 number of shape and expression components in flame model
        self.add_teeth,self.add_mouth_interior=True,True
        
        self.deform_by_point_blendshape=True # point deformed by personal blendshape(stored in points' attribute)
        self.with_param_net_smirk=True#image to expression param
        self.with_defer_render_envbrdf=True # env cube map brdf
        
        self.max_reflectance=0.15
        self.min_reflectance=0.04 #0.04
        self.max_roughness=1.0
        self.min_roughness=0.5
        self.diffuse_resolution=32*0.5
        self.specular_resolution=32*1.0
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.epochs = 0
        self.iterations=100000  # baseline: 50k
        self.position_lr_init = 0.0005
        self.position_lr_final = 0.000005
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_tv_roughness=2e-2


        self.densification_interval = 500 #100 2000
        self.opacity_reset_interval = 3000#3000
        self.densify_from_iter = 1000#10_000#500
        self.densify_until_iter = 30_000#15_000 40_000
        self.densify_grad_threshold = 0.0002*1.5
        self.warm_up_iter=1500#3000
        
        self.random_background = False
        self.with_tv_roughness=True
        self.with_envmap_consist=False
        
        
        self.lambda_normal=1e-2
        self.lambda_normal_consist=1e-5
        self.lambda_jaw_pose=0.1

        self.lambda_intrinsic_albedo=0.25
        self.lambda_intrinsic_specular=0.25
        self.lambda_envmap_consist=1e-5

        self.flame_scale_lr=1e-5
        self.flame_params_net_smirk_encoder_lr=5e-5
        self.flame_params_net_smirk_decoder_lr=5e-5

        
        self.roughness_lr = 5e-3
        self.reflectance_lr=5e-3
        self.albedo_lr=5e-3
        self.appear_lr_decay=1e-2
        
        self.envmap_lr= 5e-2 #5e-2
        self.env_map_lr_decay=1e-2

        self.shape_param_lr=1e-4
        self.shape_dirs_lr=1e-6 
        self.expression_dirs_lr=1e-6
        self.pose_dirs_lr=1e-6
        self.lbs_weights_lr=1e-4
        
        self.with_depth_supervise=True
        self.with_intrinsic_supervise=True
        self.detach_normal_refer=False

        # Hyperparameters for albedo modulation
        self.enable_heartbeat_albedo = False
        self.baseline_load_epoch = -1
        self.fps = -1

        self.heartbeat_mode3_raw_deltas_init = 0.009 
        self.heartbeat_mode3_raw_deltas_final = 0.0003 
        self.heartbeat_phase_lr_3_init = 0.002  
        self.heartbeat_phase_lr_3_final = 0.0005
        self.heartbeat_mlp_lr_3_init = 0.0038 
        self.heartbeat_mlp_lr_3_final = 0.0005 
        
        self.heartbeat_B_lr_3 = 0.0001 
        self.heartbeat_A_lr_3 = 0.00009
        self.heartbeat_A1_lr_3_init = 0.0003 
        self.heartbeat_A2_lr_3_init = 0.0003 
        self.heartbeat_mu1_lr_3_init = 0.0008 
        self.heartbeat_mu2_lr_3_init = 0.001 
        self.heartbeat_log_sig1_lr_3_init = 0.0006 
        self.heartbeat_log_sig2_lr_3_init = 0.002 

        self.heartbeat_sv_lr_3_init = 0.001

        self.heartbeat_mlp_max_steps = 80_000 
        self.heartbeat_fp_max_steps = 60_000

        self.lambda_gt_ppg = 1
        self.lambda_ppg_phase = 1
        self.lambda_ppg_pearson = 1

        self.lambda_sigma_reg = 0.6
        self.lambda_A_reg = 0.6 #0.6
        self.lambda_mu_reg = 0.1
        self.lambda_pos_weak = 0.3 #0.5
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser,model=None):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy() #args_cmdline args_cfgfile
    for k,v in vars(args_cmdline).items():
        if v != None and k not in merged_dict.keys():
            merged_dict[k] = v
    if model is not None:
        for k,v in vars(model).items():
            if k not in merged_dict.keys() and not k.startswith("_"):
                merged_dict[k]=v

    return Namespace(**merged_dict)


def add_more_argument(parser):
    parser.add_argument("--render_and_eval", action='store_true', default = True,
                        help="Rendering train and test set and evaluate the results after training")
    parser.add_argument("--test_set_ratio", type=float, default = 0.30)
    parser.add_argument("--test_set_num", type=int, default = -1) 
    parser.add_argument("--other_train_set_path", type=str,nargs="+", default = "")
    parser.add_argument("--test_set_path", type=str,nargs="+", default = "")
    parser.add_argument("--non_mouth_interior", action='store_true', default = False,)
    parser.add_argument("--nersemble_id",type=str, default = "",)

    return parser


def init_args(args):
    
    if args.with_param_net_smirk:
        print(f"smirk expression dim better be 50, expr_dim:{args.n_expr}")
        args.flame_params_net_params={
            "exp_dim":args.n_expr
        }
        
    if args.deform_by_point_blendshape:
        args.laplacian_K=5
        args.blendshape_K=3
        
    if args.with_defer_render_envbrdf:
        args.fresnel_term=0.04
        args.mip_level=3
        
    return args
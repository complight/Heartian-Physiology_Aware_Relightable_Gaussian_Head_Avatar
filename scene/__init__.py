

import os
import json
import torch
import numpy as np  
from utils.system_utils import searchForMaxIteration

from scene.gaussian_head_model import GaussianHeadModel
from arguments import ModelParams
from utils.sh_utils import SH2RGB
from utils.graphics_utils import FlamePointCloud
from utils.mesh_utils import calculate_mesh_normals,sample_initial_points_from_mesh,calculate_mesh_avg_edge_length
from scene.flame import load_flame_mesh
from typing import NamedTuple
from copy import deepcopy

class Head_Scene:
    gaussians : GaussianHeadModel
    def __init__(self, all_args,gaussians:GaussianHeadModel, load_epoch=None,dataset=None):
        self.model_path = all_args.model_path
        self.loaded_epoch = None
        self.gaussians = gaussians
        # gaussians.scene_name=(all_args.source_path).split(os.path.sep)[-1]
        gaussians.scene_name = os.path.basename(all_args.source_path)
        head_scene_info=load_head_info(all_args)
        head_scene_info.flame_scale=dataset.flame_scale
        
        if load_epoch:
            if load_epoch == -1:
                self.loaded_epoch = searchForMaxIteration(os.path.join(self.model_path, "saved_model"))
            else:
                self.loaded_epoch = load_epoch
            print("Loading trained model at epoch {}".format(self.loaded_epoch))
        
        w2c=np.zeros([4,4],dtype=np.float32)
        
        w2c[:3,:3],w2c[:3,3],w2c[3,3]=dataset[0].R.cpu().numpy(),dataset[0].T.cpu().numpy(),1.0
        c2w=np.linalg.inv(w2c)
        translate=-c2w[:3,3]
        radius=calculate_mesh_avg_edge_length(head_scene_info.flame_mesh)*dataset.flame_scale
        nerf_normalization={"radius":radius,"translate":translate}
        self.cameras_extent=nerf_normalization["radius"]
        
        if self.loaded_epoch:
            self.gaussians.create_from_pcd(head_scene_info, self.cameras_extent,
                                           dataset[0].shape_code.cuda(gaussians.device),
                                           )
            self.gaussians.load_model(os.path.join(self.model_path,
                                                           "saved_model",
                                                           "epoch_" + str(self.loaded_epoch)
                                                           ))
        else:
            self.gaussians.create_from_pcd(head_scene_info, self.cameras_extent,
                                           dataset[0].shape_code.cuda(gaussians.device),)
        
        
    def save(self, epoch):
        point_cloud_path = os.path.join(self.model_path, "saved_model/epoch_{}".format(epoch))
        self.gaussians.save_model(os.path.join(point_cloud_path))
        


class Head_SceneInfo():
    point_cloud: FlamePointCloud
    flame_mesh: dict
    flame_scale: float
    def __init__(self, point_cloud, flame_mesh, flame_scale):
        self.point_cloud = point_cloud
        self.flame_mesh = flame_mesh
        self.flame_scale = flame_scale
    
def load_head_info(args):
    flame_mesh=load_flame_mesh(args.n_shape,args.n_expr,args.add_teeth,args.add_mouth_interior)
    flame_mesh["lbs_weights"]=torch.cat([flame_mesh["lbs_weights"],torch.zeros_like(flame_mesh["lbs_weights"][:,0]).unsqueeze(1)],dim=1)
    _,flame_mesh["v_normal"]=calculate_mesh_normals(flame_mesh)
    flame_mesh["flame_triangles"]=deepcopy(flame_mesh["triangles"])
    shape_dirs,expr_dirs,pose_dirs,lbs_weights,r_eyelid_dirs,l_eyelid_dirs=None,None,None,None,None,None
    normals=None
    
    
    xyz,shape_dirs,expr_dirs,pose_dirs,lbs_weights,r_eyelid_dirs,l_eyelid_dirs,normals=\
    sample_initial_points_from_mesh(flame_mesh,num_pts_sample_fmesh=8,along_normal_scale=0.0)
    
    shs = torch.rand((xyz.shape[0], 3)) / 255.0
    pcd = FlamePointCloud(points=xyz, colors=SH2RGB(shs),normals=torch.zeros((xyz.shape[0], 3)),
                          shape_dirs=shape_dirs,expression_dirs=expr_dirs,pose_dirs=pose_dirs,
                          lbs_weights=lbs_weights,r_eyelid_dirs=r_eyelid_dirs,l_eyelid_dirs=l_eyelid_dirs,
                          normal=normals)
    
    head_scene_info = Head_SceneInfo(point_cloud=pcd,flame_mesh=flame_mesh,flame_scale=1.0)
    
    return head_scene_info
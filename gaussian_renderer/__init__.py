"""
Copyright (c) 2025 Eastbean Zhang. All rights reserved.

This code is a modified version of the original work from:
https://github.com/graphdeco-inria/gaussian-splatting

"""

import torch
import math
from scene.gaussian_head_model import GaussianHeadModel
from utils.general_utils import sample_camera_rays
from diff_gaussian_rasterization_c10 import GaussianRasterizationSettings, GaussianRasterizer
atri_bg=[0.0 for i in range(7)]


def render_with_deferred(viewpoint_camera_param, pc : GaussianHeadModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,
           iteration=torch.inf,other_cam_param=None,other_envmap=None, real_bg=None):
    
    # Update timestep for albedo modulation calculation
    pc.current_timestep = viewpoint_camera_param.timestep

    ori_camera_center=viewpoint_camera_param.camera_center.cuda(pc.device)
    
    if other_cam_param is not None:
        viewpoint_camera_param=other_cam_param
    camera_center=viewpoint_camera_param.camera_center.cuda(pc.device)
    
    shape_param,expression_param,full_pose_param=viewpoint_camera_param.shape_code.cuda(pc.device),\
        viewpoint_camera_param.exp_code.cuda(pc.device),viewpoint_camera_param.full_pose_code.cuda(pc.device)
        
    eyelid_param,translation_param=None,None
    if viewpoint_camera_param.eyelid_code is not None:
        eyelid_param=viewpoint_camera_param.eyelid_code.cuda(pc.device)
    if viewpoint_camera_param.translation_code is not None:
        translation_param=viewpoint_camera_param.translation_code.cuda(pc.device)

    warped_image=None
    if viewpoint_camera_param.warped_image is not None:
        warped_image=viewpoint_camera_param.warped_image.cuda(pc.device)
    
        
    pc.forward(shape_param,expression_param,full_pose_param,camera_center,eyelid_param,translation_param,warped_image,iteration)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera_param.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera_param.FoVy * 0.5)
    image_height=viewpoint_camera_param.image_height
    image_width=viewpoint_camera_param.image_width


    raster_settings = GaussianRasterizationSettings(
        image_height=int(image_height),
        image_width=int(image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.cat([bg_color,torch.tensor(atri_bg,device=pc.device)]),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera_param.world_view_transform.cuda(pc.device),
        projmatrix=viewpoint_camera_param.full_proj_transform.cuda(pc.device),
        sh_degree=pc.active_sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=pipe.debug
        )
    
    rasterizer=GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_deform_xyz
    means2D = screenspace_points
    opacity = pc.get_deform_opacity


    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_deform_scaling
        rotations = pc.get_deform_rotation


    shs = None
    colors_precomp = override_color
    render_results={}
    normals = pc.get_deform_normal(camera_center)
    
    roughness = pc.get_roughness
    albedo_color=pc.get_albedo
    reflectance=pc.get_reflectance

    alpha=torch.ones_like(pc._xyz[:,0],device=pc.device).unsqueeze(1)
    input_ts = torch.cat([albedo_color, normals, roughness,reflectance,alpha], dim=-1)
    
    p_hom = torch.cat([means3D, torch.ones_like(means3D[...,:1])], -1).unsqueeze(-1)
    p_view = torch.matmul(viewpoint_camera_param.world_view_transform.cuda(pc.device).transpose(0,1), p_hom)
    p_view = p_view[...,:3,:]
    depth = p_view.squeeze()[...,2:3]
    input_ts = torch.cat([input_ts,depth], dim=-1)

    out_ts, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = input_ts,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    albedo_map = out_ts[:3,...] 
    roughness_map = out_ts[6:7,...]
    reflectance_map = out_ts[7:8,...]
    normal_map = out_ts[3:6,...] 
    alpha_map=out_ts[8:9,...] 
    normal_map=normal_map.permute(1,2,0)

    if real_bg is not None:
        final_bg = real_bg.to(pc.device)
    elif bg_color is not None:
        final_bg = bg_color[:, None, None]

    if iteration>pc.warm_up_iter:
        normal_map_norm = normal_map / (torch.norm(normal_map, dim=-1, keepdim=True)+1e-6)
        HWK=(image_height,image_width,viewpoint_camera_param.K.clone().detach().cpu().numpy())
        c2p_map=sample_camera_rays(HWK,torch.tensor(viewpoint_camera_param.R,device=pc.device,dtype=torch.float32), torch.tensor(viewpoint_camera_param.T,device=pc.device,dtype=torch.float32))
        
        zero_bg = torch.zeros(3, device=pc.device)
        if other_envmap is None:
            shading_dict=pc.Envmap.shading(albedo_map,roughness_map,reflectance_map,normal_map_norm,-c2p_map,alpha_map,zero_bg)
        else:
            shading_dict=other_envmap.shading(albedo_map,roughness_map,reflectance_map,normal_map_norm,-c2p_map,alpha_map,zero_bg)
        rendered_image=shading_dict["shading"]
        render_results.update(shading_dict)

        render_results.update({"fresnel_reflectance":reflectance_map,})

        rendered_image = rendered_image + final_bg * (1.0 - alpha_map)

    else: rendered_image=albedo_map


    render_results.update(
        {"render": rendered_image,"albedo":albedo_map,
            "roughness":roughness_map,"alpha":alpha_map,
        "viewspace_points": screenspace_points,"normal":normal_map.permute(2,0,1),
        "visibility_filter" : radii > 0,
        "radii": radii}
    )
    depth_map=out_ts[9:10,...]
    depth_map=(depth_map / alpha_map)
    depth_map = torch.nan_to_num(depth_map, 0, 0)
    render_results.update({"depth":depth_map})

    
    return render_results
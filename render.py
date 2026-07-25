
import torch
from scene import Head_Scene
import os, imageio
from tqdm import tqdm
from os import makedirs
import gaussian_renderer
import torchvision
from utils.general_utils import safe_state,to8b,save_image_L

from utils.camera_utils import generate_multi_view_poses
from utils.graphics_utils import normal_from_depth_image
from scene.data_loader import TrackedData
from net_modules.NVDIFFREC.util import cubemap_to_latlong

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args,add_more_argument
from gaussian_renderer import GaussianHeadModel
import numpy as np
from net_modules.NVDIFFREC.envmap import EnvironmentMap_relight
from copy import deepcopy
from natsort import natsorted

from rppg_extractor import extract_rppg_from_rendered_images
import scipy.io as sio
import cv2

def render_set(model_path, name, epoch, cam_params, gaussians, pipeline, background,args,background_path):
    render=gaussian_renderer.render_with_deferred
    
    device=gaussians.device
    render_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(epoch), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if "render_albedo"in args.__dict__.keys() and args.render_albedo:
        render_albedo_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders_albedo")
        makedirs(render_albedo_path, exist_ok=True)
    if "render_irradiance"in args.__dict__.keys() and args.render_irradiance:
        render_irradiance_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders_irradiance")
        makedirs(render_irradiance_path, exist_ok=True)
    if  "render_specular"in args.__dict__.keys() and args.render_specular:
        render_I_specular_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders_I_specular")
        makedirs(render_I_specular_path, exist_ok=True)
        #render_specular_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders_specular")
        #render_specular_int_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders_specular_intensity")
        # makedirs(render_specular_path, exist_ok=True)
        # makedirs(render_specular_int_path, exist_ok=True)
    if "render_normal"in args.__dict__.keys() and args.render_normal:
        render_normal_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders_normal")
        makedirs(render_normal_path, exist_ok=True)
    if "render_KS"in args.__dict__.keys() and args.render_KS:
        render_KS_path = os.path.join(model_path, name, "ours_{}".format(epoch), "renders_KS")
        makedirs(render_KS_path, exist_ok=True)
    if "render_roughness"in args.__dict__.keys() and args.render_roughness:
        render_roughness_path = os.path.join(model_path, name, "ours_{}".format(epoch), "render_roughness")
        makedirs(render_roughness_path, exist_ok=True)
    if "render_reflectance"in args.__dict__.keys() and args.render_reflectance:
        render_reflectance_path = os.path.join(model_path, name, "ours_{}".format(epoch), "render_reflectance")
        makedirs(render_reflectance_path, exist_ok=True)
    if "render_depth"in args.__dict__.keys() and args.render_depth:
        render_depth_path = os.path.join(model_path, name, "ours_{}".format(epoch), "render_depth")
        makedirs(render_depth_path, exist_ok=True)
    if "render_depth_normal"in args.__dict__.keys() and args.render_depth_normal:
        render_depth_normal_path = os.path.join(model_path, name, "ours_{}".format(epoch), "render_depth_normal")
        makedirs(render_depth_normal_path, exist_ok=True)

    real_bg_dict = None
    if background_path is not None:
        needed_indices = [c.timestep for c in cam_params]
        real_bg_dict = load_real_backgrounds_by_indices(background_path, needed_indices)
        print(f"Load real background: {len(real_bg_dict)} images")

    scene_name=gaussians.scene_name
    vedio_path = os.path.dirname(render_path)
    rendering_imgs = []
    for idx, cam_param in enumerate(tqdm(cam_params, desc="Rendering progress")):
        gt = cam_param.original_image[:3, :, :]
        gt_alpha = cam_param.gt_alpha_mask.to(gt.device)
        # gt=torch.cat([gt,gt_alpha],dim=0)

        real_bg = None
        if real_bg_dict is not None:
            real_bg = real_bg_dict[cam_param.timestep].to(device)

        rendering_results = render(cam_param, gaussians, pipeline, background, real_bg=real_bg)
        rendering=rendering_results["render"]
  
        rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        alpha=rendering_results["alpha"]

        # Extract albedo modulation from all Gaussians
        if "enable_rppg" in args.__dict__.keys() and args.enable_rppg:
            with torch.no_grad():
                face_albedo = gaussians.get_albedo 
                face_G = face_albedo[:, 1].mean().item()
                if not hasattr(gaussians, '_gaussian_G_log'):
                    gaussians._gaussian_G_log = []
                gaussians._gaussian_G_log.append(face_G)

        if "render_albedo"in args.__dict__.keys() and args.render_albedo:
            rendering_albedo=rendering_results["albedo"]
            rendering_albedo=torch.cat([rendering_albedo,alpha],dim=0)
            torchvision.utils.save_image(rendering_albedo, os.path.join(render_albedo_path, '{0:05d}'.format(idx) + ".png"))
        if "render_normal"in args.__dict__.keys() and args.render_normal:
            rendering_normal=(rendering_results["normal"]+1)/2
            rendering_normal=torch.cat([rendering_normal,alpha],dim=0)
            torchvision.utils.save_image(rendering_normal, os.path.join(render_normal_path, '{0:05d}'.format(idx) + ".png"))
        if "render_irradiance"in args.__dict__.keys() and args.render_irradiance:
            rendering_irradiance=rendering_results["irradiance"]
            rendering_irradiance=torch.cat([rendering_irradiance,alpha],dim=0)
            torchvision.utils.save_image(rendering_irradiance, os.path.join(render_irradiance_path, '{0:05d}'.format(idx) + ".png"))
        if  "render_specular"in args.__dict__.keys() and args.render_specular:
            rendering_i_specular=rendering_results["I_specular"]
            rendering_i_specular=torch.cat([rendering_i_specular,alpha],dim=0)
            torchvision.utils.save_image(rendering_i_specular, os.path.join(render_I_specular_path, '{0:05d}'.format(idx) + ".png"))
            # rendering_specular=rendering_results["specular"]
            # rendering_specular_int=rendering_results["specular_intensity"]
            # rendering_specular=torch.cat([rendering_specular,alpha],dim=0)
            # rendering_specular_int=torch.cat([rendering_specular_int,alpha],dim=0)
            # torchvision.utils.save_image(rendering_specular, os.path.join(render_specular_path, '{0:05d}'.format(idx) + ".png"))
            # torchvision.utils.save_image(rendering_specular_int, os.path.join(render_specular_int_path, '{0:05d}'.format(idx) + ".png"))
            
            
        if "render_KS"in args.__dict__.keys() and args.render_KS:
            rendering_KS=rendering_results["KS"]
            rendering_KS=torch.cat([rendering_KS.repeat(3,1,1),alpha],dim=0)
            #save_image_L(rendering_KS, os.path.join(render_KS_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(rendering_KS, os.path.join(render_KS_path, '{0:05d}'.format(idx) + ".png"))
        if "render_roughness"in args.__dict__.keys() and args.render_roughness:
            rendering_roughness=rendering_results["roughness"]
            rendering_roughness=torch.cat([rendering_roughness.repeat(3,1,1),alpha],dim=0)
            torchvision.utils.save_image(rendering_roughness, os.path.join(render_roughness_path, '{0:05d}'.format(idx) + ".png"))
        if "render_reflectance"in args.__dict__.keys() and args.render_reflectance:
            render_reflectance=rendering_results["fresnel_reflectance"]
            render_reflectance=torch.cat([render_reflectance.repeat(3,1,1),alpha],dim=0)
            torchvision.utils.save_image(render_reflectance, os.path.join(render_reflectance_path, '{0:05d}'.format(idx) + ".png"))
        if "render_depth"in args.__dict__.keys() and args.render_depth:
            render_depth=rendering_results["depth"]
            render_depth=(render_depth-render_depth.min())/(render_depth.max()-render_depth.min())
            render_depth=torch.cat([render_depth.repeat(3,1,1),alpha],dim=0)
            torchvision.utils.save_image(render_depth, os.path.join(render_depth_path, '{0:05d}'.format(idx) + ".png"))
        if "render_depth_normal"in args.__dict__.keys() and args.render_depth_normal:
            render_depth=rendering_results["depth"]
            intrinsic_matrix, extrinsic_matrix = cam_param.intrinsic_matrix,cam_param.extrinsic_matrix
            normal_refer = normal_from_depth_image(render_depth[0], intrinsic_matrix.to(device), extrinsic_matrix.to(device)).permute(2,0,1)
            normal_refer=(normal_refer+1)/2
            render_depth_normal=(normal_refer*(rendering_results["alpha"].detach()))
            render_depth_normal=torch.cat([render_depth_normal,alpha],dim=0)
            torchvision.utils.save_image(render_depth_normal, os.path.join(render_depth_normal_path, '{0:05d}'.format(idx) + ".png"))
            
    rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
    print(f'{name}_{scene_name}_video.mp4')
    imageio.mimwrite(os.path.join(vedio_path, f'{name}_{scene_name}_video.mp4'), rendering_imgs, fps=gaussians.fps, quality=8)
    return rendering_imgs
    
def test_rendering_speed(model_path, cam_params, gaussians, pipeline, background,args):
    import copy,time,json
    render=gaussian_renderer.render_with_deferred
    
    savefolder=model_path
    speed_dict={}
    print("test rendering speed.......")
    cam_params=copy.deepcopy(cam_params)
    test_length=500
    test_cam_params=cam_params[:test_length]
    rendering_results = render(cam_params[-1], gaussians, pipeline, background, real_bg=None)
    start_time=time.time()
    params_extract_time=0
    for idx in tqdm(range(test_length)):
        rendering_results = render(test_cam_params[idx], gaussians, pipeline, background, real_bg=None)
        params_extract_time+=gaussians.params_extract_time
    end_time=time.time()
    avg_total_speed=(end_time-start_time)/test_length
    avg_params_extract_speed=(params_extract_time)/test_length
    speed_dict["total_speed"]={"times(ms/image)":avg_total_speed*1000,"FPS(images/s)":1.0/avg_total_speed}
    speed_dict["params_extract_speed"]={"times(ms/image)":avg_params_extract_speed*1000,"FPS(images/s)":1.0/avg_params_extract_speed}
    speed_dict["rendering_speed"]={"times(ms/image)":(avg_total_speed-avg_params_extract_speed)*1000,"FPS(images/s)":1.0/(avg_total_speed-avg_params_extract_speed)}
    


    if len(args.envmap_path)!=0:
        print("test rendering speed for relighting.......")
        envpath=args.envmap_path[0]
        new_Envmap=EnvironmentMap_relight(envpath)
        rendering_results = render(test_cam_params[-1], gaussians, pipeline, background,other_envmap=new_Envmap)
        params_extract_time=0
        start_time=time.time()
        for idx in tqdm(range(test_length)):
            cam_param=test_cam_params[idx]
            rendering_results = render(cam_param, gaussians, pipeline, background,other_envmap=new_Envmap)
            rendering=rendering_results["render"]
            params_extract_time+=gaussians.params_extract_time
        end_time=time.time()
        avg_total_speed=(end_time-start_time)/test_length
        avg_params_extract_speed=(params_extract_time)/test_length

        speed_dict["religting_total_speed"]={"times(ms/image)":avg_total_speed*1000,"FPS(images/s)":1.0/avg_total_speed}
        speed_dict["religting_params_extract_speed"]={"times(ms/image)":avg_params_extract_speed*1000,"FPS(images/s)":1.0/avg_params_extract_speed}
        speed_dict["religting_rendering_speed"]={"times(ms/image)":(avg_total_speed-avg_params_extract_speed)*1000,"FPS(images/s)":1.0/(avg_total_speed-avg_params_extract_speed)}

    with open(os.path.join(savefolder, "rendering_speed" + '.json'), 'w') as fp:
        json.dump(speed_dict, fp)
    print(speed_dict)

def render_envmap(model_path, epoch, gaussians):
    envmap_path = os.path.join(model_path, "train", "ours_{}".format(epoch))
    makedirs(envmap_path, exist_ok=True)

 
    actfun=gaussians.Envmap.activation
    diifuse_map=cubemap_to_latlong(gaussians.Envmap.diffuse,[64,64*2])
    specular_map=cubemap_to_latlong(gaussians.Envmap.specular,[64,64*2])
    diifuse_map,specular_map=actfun(diifuse_map),actfun(specular_map)
    torchvision.utils.save_image(diifuse_map.permute(2,0,1), os.path.join(envmap_path,  "diifuse_envmap.png"))
    torchvision.utils.save_image(specular_map.permute(2,0,1), os.path.join(envmap_path,  "specular_envmap.png"))
    
    diifuse_map=cubemap_to_latlong(gaussians.Envmap.diffuse,[64,64*2],150)
    specular_map=cubemap_to_latlong(gaussians.Envmap.specular,[64,64*2],150)
    diifuse_map,specular_map=actfun(diifuse_map),actfun(specular_map)
    diifuse_map,specular_map=torch.flip(diifuse_map,dims=[1]),torch.flip(specular_map,dims=[1])
    torchvision.utils.save_image(diifuse_map.permute(2,0,1), os.path.join(envmap_path,  "diifuse_envmap_shift.png"))
    torchvision.utils.save_image(specular_map.permute(2,0,1), os.path.join(envmap_path,  "specular_envmap_shift.png"))

def render_relighting(model_path, name, epoch, cam_params, gaussians, pipeline, background,args):
    import time
    print("rendering relighting...")
    render=gaussian_renderer.render_with_deferred
        

    if len(args.envmap_path)!=0:
        for envpath in args.envmap_path:

            new_Envmap=EnvironmentMap_relight(envpath)
            max_steps=min(len(cam_params),500)
            envmap_name=os.path.basename(envpath)
            render_path = os.path.join(model_path, "dynamic_relight",f"{name}_{epoch}", f"renders_relight-{envmap_name}")
            makedirs(render_path, exist_ok=True)
            vedio_path = os.path.dirname(render_path)
            scene_name=gaussians.scene_name
            rendering_imgs = []
            for idx, cam_param in enumerate(tqdm(cam_params, desc="Rendering progress")):
                rotate_y_degree=idx/max_steps*360
                new_Envmap.set_rotate_y(rotate_y_degree)
                rendering_results = render(cam_param, gaussians, pipeline, background,other_envmap=new_Envmap)
                rendering=rendering_results["render"]
                if args.with_relight_background:
                    back_map=new_Envmap.get_back_map(rotate_y_degree)
                    #alpha=rendering_results["alpha"]
                    alpha=cam_param.gt_alpha_mask.cuda(rendering.device)
                    rendering=rendering*alpha+(1-alpha)*back_map
                # rendering=torch.cat([rendering,alpha.to(rendering.device)],dim=0)
                rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
                torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
            
            if 'UBFC-rPPG' in args.source_path:
                output_avi_path = os.path.join(args.source_path, f'vid_ours_relit.avi')
                imageio.mimwrite(output_avi_path, rendering_imgs, fps=gaussians.fps, codec='rawvideo',  quality=None)
                print(f"Saved rendered '.avi' video to {output_avi_path}")
            elif 'PURE' in args.source_path:
                scene_name = os.path.basename(args.source_path)
                output_dir = os.path.join(args.source_path, f'{scene_name}_ours_relit')
                os.makedirs(output_dir, exist_ok=True)

                for i, img in enumerate(rendering_imgs):
                    cv2.imwrite(os.path.join(output_dir, f'{i:05d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"Saved rendered frames to {output_dir}")
            
            # save relit video
            imageio.mimwrite(os.path.join(vedio_path, f'{name}_{scene_name}_relight-{envmap_name}_video.mp4'), rendering_imgs, fps=gaussians.fps, quality=8)

def render_static_relighting(model_path, name,epoch, cam_params, gaussians, pipeline, background,args,frame_idxs=[]):
    import time
    print(f"rendering static relighting. for {name} set..")
    render=gaussian_renderer.render_with_deferred

    if len(args.envmap_path)!=0:
        for envpath in args.envmap_path:
            new_Envmap=EnvironmentMap_relight(envpath)
            max_steps=360#180
            envmap_name=os.path.basename(envpath)
            for frame_idx in tqdm(frame_idxs):
                cam_param=cam_params[frame_idx]
                render_path = os.path.join(model_path, "static_relight",f"{name}_{epoch}",  f"renders_relight_{envmap_name}_{frame_idx}")
                makedirs(render_path, exist_ok=True)
                vedio_path = os.path.dirname(render_path)
                scene_name=gaussians.scene_name
                rendering_imgs = []
                for idx in (tqdm(range(max_steps), desc="Rendering progress")):
                    rotate_y_degree=idx/max_steps*360
                    new_Envmap.set_rotate_y(rotate_y_degree)
                    rendering_results = render(cam_param, gaussians, pipeline, background,other_envmap=new_Envmap)
                    rendering=rendering_results["render"]
                    alpha=rendering_results["alpha"]
                    if args.with_relight_background:
                        back_map=new_Envmap.get_back_map(rotate_y_degree)
                        #alpha=rendering_results["alpha"]
                        alpha=cam_param.gt_alpha_mask.cuda(rendering.device)
                        rendering=rendering*alpha+(1-alpha)*back_map
                    else:
                        rendering=torch.cat([rendering,alpha.to(rendering.device)],dim=0)
                    rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
                    torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
                imageio.mimwrite(os.path.join(vedio_path, f'{name}_{scene_name}_relight_{envmap_name}_{frame_idx}_video.mp4'), rendering_imgs, fps=gaussians.fps, quality=8)

def render_material_editing(model_path, name, epoch, cam_params, gaussians, pipeline, background,args):
    import time
    render=gaussian_renderer.render_with_deferred
    
    if len(args.envmap_path)>0:
        new_Envmap=EnvironmentMap_relight(args.envmap_path[0])
    
    max_steps=len(cam_params)
    # envmap_name=os.path.basename(args.envmap_path[0])
    render_path = os.path.join(model_path, name, "ours_{}".format(epoch), f"renders_material_editing")
    makedirs(render_path, exist_ok=True)
    vedio_path = os.path.dirname(render_path)
    scene_name=gaussians.scene_name
    rendering_imgs = []
    
    for idx, cam_param in enumerate(tqdm(cam_params, desc="Rendering progress")):
        roughness_scale=(1-idx/max_steps)
        if  args.envmap_path is not None:
            new_Envmap.set_reflectance_scale(idx/max_steps)
            new_Envmap.set_roughness_scale(roughness_scale)
            new_Envmap.set_rotate_y(idx/max_steps*360)
            rendering_results = render(cam_param, gaussians, pipeline, background,other_envmap=new_Envmap)
            
        else:
            gaussians.Envmap.set_reflectance(idx/max_steps)
            gaussians.Envmap.set_reflectance_scale(idx/max_steps)
            gaussians.Envmap.set_roughness_scale(roughness_scale)
            rendering_results = render(cam_param, gaussians, pipeline, background)
            
        rendering=rendering_results["render"]

        #alpha=rendering_results["alpha"]
        #rendering=torch.cat([rendering,alpha.to(rendering.device)],dim=0)
        rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(vedio_path, f'{name}_{scene_name}_material_editing_video.mp4'), rendering_imgs, fps=gaussians.fps, quality=8)
    gaussians.Envmap.set_reflectance_scale(0.0)
    gaussians.Envmap.set_roughness_scale(1.0)

def render_static_material_editing(model_path, name, epoch, cam_params, gaussians, pipeline, background,args,frame_idxs=[]):
    import time
    render=gaussian_renderer.render_with_deferred
    
    if len(args.envmap_path)>0:
        new_Envmap=EnvironmentMap_relight(args.envmap_path[0])
        envmap_name=os.path.basename(args.envmap_path[0])

    max_steps=240#100
    
    for frame_idx in frame_idxs:
        render_path = os.path.join(model_path, "static_material_editing", f"{name}_{epoch}", f"material_editing_{frame_idx}")
        makedirs(render_path, exist_ok=True)
        vedio_path = os.path.dirname(render_path)
        scene_name=gaussians.scene_name
        rendering_imgs = []
        cam_param=cam_params[frame_idx]
        for idx in (tqdm(range(max_steps), desc="Rendering progress")):
            roughness_scale=(1-idx/max_steps)
            if  args.envmap_path is not None:
                new_Envmap.set_reflectance_scale(idx/max_steps)
                new_Envmap.set_roughness_scale(roughness_scale)
                #new_Envmap.set_rotate_y(idx/max_steps*360)
                rendering_results = render(cam_param, gaussians, pipeline, background,other_envmap=new_Envmap)
                
            else:
                gaussians.Envmap.set_reflectance(idx/max_steps)
                gaussians.Envmap.set_reflectance_scale(idx/max_steps)
                gaussians.Envmap.set_roughness_scale(roughness_scale)
                rendering_results = render(cam_param, gaussians, pipeline, background)
                
            rendering=rendering_results["render"]
            alpha=rendering_results["alpha"]
            if args.with_relight_background and len(args.envmap_path)>0:
                back_map=new_Envmap.get_back_map(0.0)
                alpha=rendering_results["alpha"]
                #alpha=cam_param.gt_alpha_mask.cuda(rendering.device)
                rendering=rendering*alpha+(1-alpha)*back_map
            # rendering=torch.cat([rendering,alpha.to(rendering.device)],dim=0)
            rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
        imageio.mimwrite(os.path.join(vedio_path, f'{name}_{scene_name}_{frame_idx}_static_material_editing.mp4'), rendering_imgs, fps=gaussians.fps, quality=8)
        gaussians.Envmap.set_reflectance_scale(0.0)
        gaussians.Envmap.set_roughness_scale(1.0)

def render_cross_reenactment(model_path,ori_params, epoch, gaussians, pipeline, background,args):
    
    render=gaussian_renderer.render_with_deferred
    
    for cross_idx,corss_path in enumerate(args.corss_source_paths):
        args_copy=deepcopy(args)
        args_copy.source_path=corss_path
        args_copy.test_set_ratio=0.0
        cross_id_name=os.path.basename(corss_path)
        print("Loading cross id data...")
        Cross_head_data=TrackedData(args_copy.source_path,args_copy,split="train",load_image=True)
        
        scene_name=gaussians.scene_name
        render_path = os.path.join(model_path, "test_cross_reenactment", "ours_{}".format(epoch),f"{cross_id_name}_reenactment_{scene_name}")
        makedirs(render_path, exist_ok=True)
        # video_path = os.path.join(model_path, "test", "ours_{}".format(epoch))
        video_path = os.path.dirname(render_path)
        makedirs(render_path, exist_ok=True)
        rendering_imgs = []
        cam_params=Cross_head_data
        ori_param=deepcopy(ori_params[0])

        for idx, cam_param in enumerate(tqdm(cam_params, desc="Rendering progress")):
            rendering_results = render(cam_param, gaussians, pipeline, background)
            rendering=rendering_results["render"]
            if "alpha" in rendering_results.keys():
                alpha=rendering_results["alpha"]
            else:
                alpha=cam_param.gt_alpha_mask
            rendering=torch.cat([rendering,alpha.to(rendering.device)],dim=0)
            rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
        imageio.mimwrite(os.path.join(video_path, f'{cross_id_name}_reenactment_{scene_name}_video.mp4'), rendering_imgs, fps=gaussians.fps, quality=8)
        
def render_multi_views(model_path, name, epoch, cam_params, gaussians, pipeline, background,multi_cam_params=None):
    if multi_cam_params is None:
        multi_cam_params=generate_multi_view_poses(cam_params)
    render=gaussian_renderer.render_with_deferred

        
    scene_name=gaussians.scene_name
    render_path = os.path.join(model_path, f"{name}", "ours_{}".format(epoch),f'{name}_multi-views_{scene_name}')
    makedirs(render_path, exist_ok=True)
    #video_path = os.path.join(model_path, name, "ours_{}".format(epoch))
    video_path = os.path.dirname(render_path)
    makedirs(video_path, exist_ok=True)
    rendering_imgs = []
    for idx, cam_param in enumerate(tqdm(cam_params, desc="Rendering progress")):
        other_cam_parm=multi_cam_params[idx]
        rendering_results=render(cam_param, gaussians, pipeline, background,other_cam_param=other_cam_parm)
        rendering = rendering_results["render"]
        if "alpha" in rendering_results.keys():
            alpha=rendering_results["alpha"]
        else:
            alpha=cam_param.gt_alpha_mask.to(rendering.device)
        rendering=torch.cat([rendering,alpha.to(rendering.device)],dim=0)
        rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(video_path, f'{name}_multi-views_{scene_name}_video.mp4'), rendering_imgs, fps=gaussians.fps, quality=8)

def render_static_multi_views(model_path, name, epoch, cam_params, gaussians, pipeline, background,frame_idxs=[]):
    render=gaussian_renderer.render
    render=gaussian_renderer.render_with_deferred
    for frame_idx in frame_idxs:
        num_keyframes=360#180
        cam_param_frame=[deepcopy(cam_params[frame_idx]) for i in range(num_keyframes)]
        multi_cam_params=generate_multi_view_poses(cam_param_frame,num_keyframes=num_keyframes,y_offset=-0.0)
   
        scene_name=gaussians.scene_name
        render_path = os.path.join(model_path, f"{name}_static_multiviews", "ours_{}".format(epoch),f"{scene_name}_{frame_idx:05d}",)
        makedirs(render_path, exist_ok=True)
        #video_path = os.path.join(model_path, name, "ours_{}".format(epoch))
        video_path = os.path.dirname(render_path)
        makedirs(video_path, exist_ok=True)
        rendering_imgs = []
        for idx, cam_param in enumerate(tqdm(multi_cam_params, desc="Rendering progress")):
            # other_cam_parm=multi_cam_params[idx] ,other_cam_param=other_cam_parm
            rendering_results=render(cam_param, gaussians, pipeline, background)
            rendering = rendering_results["render"]
            if "alpha" in rendering_results.keys():
                alpha=rendering_results["alpha"]
            else:
                alpha=cam_param.gt_alpha_mask.to(rendering.device)
            rendering=torch.cat([rendering,alpha.to(rendering.device)],dim=0)
            rendering_imgs.append(to8b(rendering.detach().cpu().numpy()))
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        rendering_imgs = np.stack(rendering_imgs, 0).transpose(0, 2, 3, 1)
        imageio.mimwrite(os.path.join(video_path, f'{name}_static_multiviews_{scene_name}_{frame_idx:05d}_video.mp4'), rendering_imgs, fps=gaussians.fps, quality=8)

def extract_optimized_params( cam_params_train,cam_params_test, gaussians,data_path):
    import json
    
    all_cam_params=cam_params_train+cam_params_test
    tracked_params_path=os.path.join(data_path,"tracked_params.json")
    with open(tracked_params_path) as json_file:
        tracked_params_dict = json.load(json_file)
    image_keys=tracked_params_dict.keys()
    image_keys=natsorted(image_keys)
    print(f"Extracting {os.path.basename(data_path)} optimized params...")
    for idx,cam_param in enumerate(tqdm(all_cam_params)):
        image_key=image_keys[idx]
        assert idx==int(image_key.split(".")[0])
        if  gaussians.with_param_net_smirk and cam_param.warped_image is not None:
            warped_image=cam_param.warped_image.cuda(gaussians.device)
            out_params=gaussians.flame_params_net(warped_image)
            expression_param = out_params['expression_params']
            jaw_params = out_params.get('jaw_params', None)
            eyelid_param = out_params.get('eyelid_params', None)
            #[(global)3, neck (0)3, (jaw)3, eyepose (0)6]
            full_pose_param=cam_param.full_pose_code.cuda(gaussians.device)
            if gaussians.use_smirk_jaw_pose:
                full_pose_param[:,6:9]=jaw_params
            tracked_params_dict[image_key]["fullposecode"]=full_pose_param.detach().cpu().numpy().tolist()
            tracked_params_dict[image_key]["eyelids"]=eyelid_param.detach().cpu().numpy().tolist()
            tracked_params_dict[image_key]["expcode"]=expression_param.detach().cpu().numpy().tolist()
    with open(os.path.join(data_path, "tracked_params_optimized" + '.json'), 'w') as fjson:
            json.dump(tracked_params_dict,fjson)
    fjson.close()
    print("Done!")

# Background Ablation: composite the real background
def load_real_backgrounds_by_indices(bg_folder, indices):
    bg_dict = {}
    unique_indices = sorted(set(indices))
    for idx in tqdm(unique_indices, desc="Loading real backgrounds"):
        bg_path = os.path.join(bg_folder, f'{idx:05d}.png')
        bg_img = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
        if bg_img is None:
            raise FileNotFoundError(f"🔧 Background image not found: {bg_path}. Please ensure the preprocessing is run with the saved temporal background.")
        bg_img = bg_img[..., :3]
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        bg_img = bg_img.astype(np.float32) / 255.0
        bg_tensor = torch.from_numpy(bg_img).permute(2, 0, 1).contiguous()

        bg_dict[idx] = bg_tensor
    return bg_dict

def render_sets(dataset_args : ModelParams, epoch : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,all_args):
    with torch.no_grad():
        gaussians = GaussianHeadModel(dataset_args.sh_degree,all_args)
        _train_dataset=TrackedData(args.source_path,args,split='train',pre_load=True)
        _test_dataset=TrackedData(args.source_path,args,split='test',pre_load=True)
        train_dataset=[_data for _data in _train_dataset]
        test_dataset=[_data for _data in _test_dataset]
        
        all_args.with_intrinsic_supervise=False
        
        scene = Head_Scene(all_args, gaussians, load_epoch=epoch,dataset=_train_dataset)
        gaussians.set_eval(True)

        bg_color= 1 if dataset_args.white_background else 0
        background = torch.tensor([bg_color]*3, dtype=torch.float32, device="cuda")

        if getattr(all_args, "with_real_bg", False):
            background_path = os.path.join(args.source_path, 'background')
            print(f"Real Background Path:{background_path}")
        else:
            background_path = None

        test_rendering_speed(dataset_args.model_path, train_dataset, gaussians, pipeline, background,all_args)
        # buffer for visualizing spatial variance
        if all_args.enable_rppg:
            gaussians._modulation_buffer = {}
        if not skip_train:
            train_imgs = render_set(dataset_args.model_path, "train", scene.loaded_epoch, train_dataset, gaussians, pipeline, background,all_args, background_path)
            if all_args.enable_rppg:    
                gaussians._gaussian_G_log_train = gaussians._gaussian_G_log.copy()
                gaussians._gaussian_G_log = [] 
        if not skip_test:
            test_imgs = render_set(dataset_args.model_path, "test", scene.loaded_epoch, test_dataset, gaussians, pipeline, background,all_args, background_path)
            if all_args.enable_rppg:
                gaussians._gaussian_G_log_test = gaussians._gaussian_G_log.copy()
                gaussians._gaussian_G_log = []
        
        # Save full rendered video with sorted indices + reorganize as the original dataset format
        if not skip_train and not skip_test:
            train_indices = _train_dataset.split_indices
            test_indices  = _test_dataset.split_indices

            all_img_pairs = list(zip(train_indices, train_imgs)) + list(zip(test_indices, test_imgs))
            
            if 'MMPD' in args.source_path:
                skipped_path = os.path.join(args.source_path, 'skipped_indices.npy')
                if os.path.exists(skipped_path):
                    skipped_indices = np.load(skipped_path)
                    if len(skipped_indices) > 0:
                        zero_frame = np.zeros_like(all_img_pairs[0][1])
                        all_img_pairs += [(int(idx), zero_frame) for idx in skipped_indices]
                        print(f"🔧 Make up the artifact frame in MMPD at{skipped_indices}")

            all_img_pairs_sorted = sorted(all_img_pairs, key=lambda x: x[0])
            full_render_imgs = np.stack([p[1] for p in all_img_pairs_sorted], 0)
            print(f"🔧 full_render_imgs shape: {full_render_imgs.shape}")

            # save rendered Heartian videos
            # imageio.mimwrite(os.path.join(args.source_path, f'full_{gaussians.scene_name}_video.mp4'),full_render_imgs,fps=gaussians.fps,quality=8)

            if 'UBFC-rPPG' in args.source_path:
                output_avi_path = os.path.join(args.source_path, f'vid_ours.avi')
                imageio.mimwrite(output_avi_path, full_render_imgs, fps=gaussians.fps, codec='rawvideo',  quality=None)
                print(f"Saved rendered '.avi' video to {output_avi_path}")

                if "render_albedo" in args.__dict__.keys() and args.render_albedo:
                    # collect albedo frames in sorted order
                    train_albedo_path = os.path.join(dataset_args.model_path, "train", f"ours_{scene.loaded_epoch}", "renders_albedo")
                    test_albedo_path = os.path.join(dataset_args.model_path, "test", f"ours_{scene.loaded_epoch}", "renders_albedo")
                    
                    train_albedo_imgs = [cv2.imread(os.path.join(train_albedo_path, f'{i:05d}.png')) for i in range(len(train_indices))]
                    test_albedo_imgs = [cv2.imread(os.path.join(test_albedo_path, f'{i:05d}.png')) for i in range(len(test_indices))]
                    
                    all_albedo_pairs = list(zip(train_indices, train_albedo_imgs)) + list(zip(test_indices, test_albedo_imgs))
                    all_albedo_sorted = sorted(all_albedo_pairs, key=lambda x: x[0])
                    full_albedo_imgs = np.stack([cv2.cvtColor(p[1][:,:,:3], cv2.COLOR_BGR2RGB) for p in all_albedo_sorted], 0)
                    
                    output_albedo_path = os.path.join(args.source_path, f'vid_ours_albedo.avi')
                    imageio.mimwrite(output_albedo_path, full_albedo_imgs, fps=gaussians.fps, codec='rawvideo', quality=None)
                    print(f"Saved albedo '.avi' video to {output_albedo_path}")

            elif 'MMPD' in args.source_path:
                # Configuration for handling memory issue
                video = np.empty(full_render_imgs.shape, dtype=np.float32)
                np.copyto(video, full_render_imgs)
                del full_render_imgs
                video /= 255.0
                    
                scene_name = os.path.basename(args.source_path)
                output_mat_path = os.path.join(args.source_path, f'{scene_name}_ours.mat')

                import h5py
                with h5py.File(output_mat_path, 'w') as f_out:
                    f_out.create_dataset('video', data=video, compression='gzip', compression_opts=4)
                print(f"Saved rendered mat to {output_mat_path}")
            elif 'PURE' in args.source_path:
                scene_name = os.path.basename(args.source_path)
                output_dir = os.path.join(args.source_path, f'{scene_name}_ours')
                os.makedirs(output_dir, exist_ok=True)

                for i, img in enumerate(full_render_imgs):
                    cv2.imwrite(os.path.join(output_dir, f'{i:05d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"Saved rendered frames to {output_dir}")


        # Evaluate embedded rPPG signal
        if all_args.enable_rppg and len(gaussians._gaussian_G_log_train) > 0 and len(gaussians._gaussian_G_log_test) > 0:

            train_indices = _train_dataset.split_indices
            test_indices  = _test_dataset.split_indices

            all_pairs = list(zip(train_indices, gaussians._gaussian_G_log_train)) + list(zip(test_indices,  gaussians._gaussian_G_log_test))
            all_pairs_sorted = sorted(all_pairs, key=lambda x: x[0])

            full_indices = [p[0] for p in all_pairs_sorted]
            full_G_log   = np.array([p[1] for p in all_pairs_sorted])

            full_output_dir = os.path.join(dataset_args.model_path, "rppg_eval_full")

            result_full, filtered_full, filtered_gt_full = extract_rppg_from_rendered_images(
                signal=full_G_log,
                split_indices=full_indices,
                image_dir=None,
                fps=gaussians.fps,
                output_dir=full_output_dir,
                source_path=all_args.source_path,
                split_name="full",
                test_indices=test_indices,
            )

            # train pointwise
            train_output_dir = os.path.join(dataset_args.model_path, "rppg_eval_train_pointwise")
            extract_rppg_from_rendered_images(
                signal=np.array(gaussians._gaussian_G_log_train),
                split_indices=train_indices,
                image_dir=None,
                fps=gaussians.fps,
                output_dir=train_output_dir,
                source_path=all_args.source_path,
                split_name="train",
                filtered_full=filtered_full,
                filtered_gt_full=filtered_gt_full,
                full_split_indices=full_indices,
            )

            # test pointwise
            test_output_dir = os.path.join(dataset_args.model_path, "rppg_eval_test_pointwise")
            extract_rppg_from_rendered_images(
                signal=np.array(gaussians._gaussian_G_log_test),
                split_indices=test_indices,
                image_dir=None,
                fps=gaussians.fps,
                output_dir=test_output_dir,
                source_path=all_args.source_path,
                split_name="test",
                filtered_full=filtered_full,
                filtered_gt_full=filtered_gt_full,
                full_split_indices=full_indices,
            )

            # visualize spatial variance
            spatial_vis_path = os.path.join(dataset_args.model_path, "rppg_eval_full", "spatial_weights.png")
            gaussians.visualize_spatial_weights(output_path=spatial_vis_path)
             
        if all_args.render_multi_views:
            render_multi_views(dataset_args.model_path, "train", scene.loaded_epoch, train_dataset, gaussians, pipeline, background)
            render_multi_views(dataset_args.model_path, "test", scene.loaded_epoch, test_dataset, gaussians, pipeline, background)
        
        if all_args.render_envmap:
            render_envmap(dataset_args.model_path,scene.loaded_epoch,gaussians)
        if all_args.render_relighting:
            train_indices = _train_dataset.split_indices
            test_indices  = _test_dataset.split_indices
            all_pairs = list(zip(train_indices, train_dataset)) + list(zip(test_indices, test_dataset))
            all_pairs_sorted = sorted(all_pairs, key=lambda x: x[0])
            full_cam_params = [p[1] for p in all_pairs_sorted]
            render_relighting(dataset_args.model_path, "full", scene.loaded_epoch, full_cam_params, gaussians, pipeline, background, all_args)
            # render_relighting(dataset_args.model_path, "test", scene.loaded_epoch, test_dataset, gaussians, pipeline, background,all_args)
            
        if all_args.render_material_editing:
            render_material_editing(dataset_args.model_path, "test", scene.loaded_epoch, test_dataset, gaussians, pipeline, background,all_args)

        if len(all_args.test_static_multiviews_idxs) >0:
            render_static_multi_views(dataset_args.model_path, "test", scene.loaded_epoch, test_dataset, gaussians, pipeline, background,all_args.test_static_multiviews_idxs)
        if len(all_args.train_static_multiviews_idxs) >0:
            render_static_multi_views(dataset_args.model_path, "train", scene.loaded_epoch, train_dataset, gaussians, pipeline, background,all_args.train_static_multiviews_idxs)
        if len(all_args.corss_source_paths)!=0:
            render_cross_reenactment(dataset_args.model_path,test_dataset, scene.loaded_epoch, gaussians, pipeline, background,all_args)
        if len(all_args.test_static_relight_idxs)>0:
            render_static_relighting(dataset_args.model_path, "test", scene.loaded_epoch, test_dataset, gaussians, pipeline, background,all_args,all_args.test_static_relight_idxs)
        if len(all_args.train_static_relight_idxs)>0:
            render_static_relighting(dataset_args.model_path, "train", scene.loaded_epoch, train_dataset, gaussians, pipeline, background,all_args,all_args.train_static_relight_idxs)
        if len(all_args.test_static_material_edting_idxs)>0:
            render_static_material_editing(dataset_args.model_path, "test", scene.loaded_epoch, test_dataset, gaussians, pipeline, background,all_args,all_args.test_static_material_edting_idxs)
        if len(all_args.train_static_material_edting_idxs)>0:
            render_static_material_editing(dataset_args.model_path, "train", scene.loaded_epoch, train_dataset, gaussians, pipeline, background,all_args,all_args.train_static_material_edting_idxs)

        if all_args.source_params_path!="":
            extract_optimized_params(train_dataset,test_dataset, gaussians,all_args.source_params_path)

    gaussians.set_eval(False)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--epoch", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_multi_views", action="store_true")
    parser.add_argument("--corss_source_paths", nargs="+",type=str,default=[],
                        help="Data Path of cross ID for reenactment")
    parser.add_argument("--param_net_ckpt_paths", nargs="+",type=str,default=[],
                        )
    parser.add_argument("--quiet", action="store_true") 

    parser.add_argument("--enable_rppg", action="store_true", default=False)
    
    parser.add_argument("--render_albedo", action="store_true",default=False)
    parser.add_argument("--render_irradiance", action="store_true",default=False)
    parser.add_argument("--render_specular", action="store_true",default=False)
    parser.add_argument("--render_normal", action="store_true",default=False)
    parser.add_argument("--render_KS", action="store_true",default=False)
    
    parser.add_argument("--render_roughness", action="store_true",default=False)
    parser.add_argument("--render_reflectance", action="store_true",default=False)
    parser.add_argument("--render_depth", action="store_true",default=False)
    parser.add_argument("--render_depth_normal", action="store_true",default=False)
    parser.add_argument("--render_envmap", action="store_true",default=False)
    parser.add_argument("--render_relighting", action="store_true",default=False)
    parser.add_argument("--render_material_editing", action="store_true",default=False)
    
    parser.add_argument("--envmap_path",nargs="+", type=str,default=[])
    parser.add_argument("--envmap_id", nargs="+",type=int,default=[0])

    parser.add_argument("--test_static_multiviews_idxs", nargs="+",type=int,default=[])
    parser.add_argument("--train_static_multiviews_idxs", nargs="+",type=int,default=[])
    parser.add_argument("--test_static_relight_idxs", nargs="+",type=int,default=[])
    parser.add_argument("--train_static_relight_idxs", nargs="+",type=int,default=[]) 
    parser.add_argument("--test_static_material_edting_idxs", nargs="+",type=int,default=[])
    parser.add_argument("--train_static_material_edting_idxs", nargs="+",type=int,default=[])
    parser.add_argument("--source_params_path",type=str,default="")
    parser.add_argument("--with_relight_background",action="store_true",default=False)

    parser.add_argument("--with_real_bg",action="store_true",default=False)
    parser=add_more_argument(parser)
    args = get_combined_args(parser,model)

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)
    
    render_sets(model.extract(args), args.epoch, pipeline.extract(args), args.skip_train, args.skip_test,args)
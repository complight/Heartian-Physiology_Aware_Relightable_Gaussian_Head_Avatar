

import os
import json
import torch
import numpy as np
import torchvision
from copy import deepcopy
from natsort import natsorted
from utils.graphics_utils import getWorld2View2, getProjectionMatrix,fov2focal,focal2fov
from utils.general_utils import run_mediapipe, crop_face
from typing import NamedTuple
from glob import glob
from PIL import Image
from tqdm import tqdm
from skimage.transform import warp

# Merry
import random


class Camera_params():
    original_image: torch.tensor
    gt_alpha_mask: torch.tensor
    albedo: torch.tensor
    specular: torch.tensor
    warped_image: torch.tensor
    
    R:torch.tensor
    T: torch.tensor
    world_view_transform: torch.tensor
    projection_matrix: torch.tensor
    full_proj_transform: torch.tensor
    camera_center: torch.tensor
    intrinsic_matrix: torch.tensor
    K: torch.tensor
    extrinsic_matrix: torch.tensor
    
    shape_code: torch.tensor
    translation_code: torch.tensor
    eyelid_code: torch.tensor
    full_pose_code: torch.tensor
    exp_code: torch.tensor
    
    FoVx : float
    FoVy: float
    image_width: int
    image_height: int
    image_name: str
    
    def __init__(self,original_image,gt_alpha_mask,albedo,specular,warped_image,world_view_transform,projection_matrix,full_proj_transform,
                 camera_center,intrinsic_matrix,K,R,T,extrinsic_matrix,shapecode,translation_code,eyelid_code,fullposecode,expcode,
                 FoVx,FoVy,image_width,image_height,image_name, timestep):
        self.original_image=original_image
        self.gt_alpha_mask=gt_alpha_mask
        self.albedo=albedo
        self.specular=specular
        self.warped_image=warped_image
        self.world_view_transform=world_view_transform
        self.projection_matrix=projection_matrix
        self.full_proj_transform=full_proj_transform
        self.camera_center=camera_center
        self.intrinsic_matrix=intrinsic_matrix
        self.K=K
        self.R=R
        self.T=T
        self.extrinsic_matrix=extrinsic_matrix
        self.shape_code=shapecode
        self.translation_code=translation_code
        self.eyelid_code=eyelid_code
        self.full_pose_code=fullposecode
        self.exp_code=expcode
        self.FoVx=FoVx
        self.FoVy=FoVy
        self.image_width=image_width
        self.image_height=image_height
        self.image_name=image_name
        self.timestep=timestep
    
class TrackedData(torch.utils.data.Dataset):
    def __init__(self, path,args,split,pre_load=True,load_image=True,device='cpu'):
        self.args=args
        self.pre_load=pre_load
        self.load_image=load_image
        self.device=device
        
        
        if os.path.isdir(path):
            images_path = os.path.join(path, "image")
            mask_prepath = os.path.join(path, "mask")
            if not os.path.exists(images_path):
                images_path = os.path.join(path, "images")
            imagepath_list = glob(images_path + '/*.jpg') + glob(images_path + '/*.png') + glob(images_path + '/*.bmp')
            print('total {} images'.format(len(imagepath_list)))
            imagepath_list = natsorted(imagepath_list)

        tracked_params_path = os.path.join(path, "tracked_params.json")
        self.flame_scale = 4.0
        if not os.path.exists(tracked_params_path):
            tracked_params_path = os.path.join(path, "tracked_params_v2.json")
            self.flame_scale = 1.0
        with open(tracked_params_path) as json_file:
            tracked_params_dict = json.load(json_file)

        train_set_len = int(len(imagepath_list) * (1 - args.test_set_ratio))
        if args.test_set_num != -1:
            train_set_len = int(len(imagepath_list) - args.test_set_num)
        
        # Merry: configuration for train/test split
        indices = list(range(len(imagepath_list)))
        random.seed(42)
        random.shuffle(indices)

        train_indices = sorted(indices[:train_set_len])
        test_indices  = sorted(indices[train_set_len:])
        print(f"🔧 Train data len: {len(train_indices)}")
        print(f"🔧 Test data len: {len(test_indices)}")

        if split == 'train':
            imagepath_list = [imagepath_list[i] for i in train_indices]
            # imagepath_list = imagepath_list[:train_set_len]
        else:
            imagepath_list = [imagepath_list[i] for i in test_indices]
            # imagepath_list = imagepath_list[train_set_len:]
        
        self.split_indices = train_indices if split == 'train' else test_indices 

        self.imagepath_list=imagepath_list
        self.data_len=len(imagepath_list)
        
        if args.with_param_net_smirk:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            base_options = python.BaseOptions(model_asset_path='./assets/smirk/face_landmarker.task')
            options = vision.FaceLandmarkerOptions(base_options=base_options,
                                                output_face_blendshapes=True,
                                                output_facial_transformation_matrixes=True,
                                                num_faces=1,
                                                min_face_detection_confidence=0.1,
                                                min_face_presence_confidence=0.1)
            self.detector = vision.FaceLandmarker.create_from_options(options)

        # assume same resolution for all images
        image = Image.open(imagepath_list[0])
        self.imagew, self.imageh = image.size[0], image.size[1]
        
        self.bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device) if args.white_background else torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        shapecode = torch.tensor(tracked_params_dict["shapecode"], device=self.device)[:, :args.n_shape]
        self.cam_params_list=[]
        for idx, imagepath in tqdm(enumerate(imagepath_list)):
            imagename = imagepath.split('/')[-1].split('.')[0]
            image_basename = os.path.basename(imagepath)
            imagekey = imagename if imagename in tracked_params_dict.keys() else image_basename
            
            if self.pre_load and self.load_image:
                image, mask_data, albedo, specular, warped_image = self._load_images(imagepath, args, self.bg)
            else:
                image, mask_data, albedo, specular, warped_image = None, None, None, None, None
            
            
            if "translation" in tracked_params_dict[imagekey].keys():
                translation_code = torch.tensor(tracked_params_dict[imagekey]["translation"], dtype=torch.float32, device=self.device)
            else:
                translation_code = None
            if "eyelids" in tracked_params_dict[imagekey].keys():
                eyelid_code = torch.tensor(tracked_params_dict[imagekey]["eyelids"], dtype=torch.float32, device=self.device)
            else:
                eyelid_code = None
                
            if  "intrinsics"in tracked_params_dict.keys():
                intrinsics=tracked_params_dict["intrinsics"]#[fx, fy, cx, cy]
                fovx=2*np.arctan2(intrinsics[2],intrinsics[0])
            
            
            fullposecode = torch.tensor(tracked_params_dict[imagekey]["fullposecode"], dtype=torch.float32, device=self.device)
            expcode = torch.tensor(tracked_params_dict[imagekey]["expcode"], dtype=torch.float32, device=self.device)[:, :args.n_expr]
            
            fovy = focal2fov(fov2focal(fovx, self.imagew), self.imageh)
            world_view_transform,projection_matrix,full_proj_transform,camera_center,extrinsic_matrix,intrinsic_matrix,R,T = \
                self._load_camera(tracked_params_dict,imagekey,fovx,fovy,zfar=100.0,znear =0.01)

            # Merry: store the timestep
            original_idx = train_indices[idx] if split == 'train' else test_indices[idx]
            self.cam_params_list.append(Camera_params(original_image=image,gt_alpha_mask=mask_data,albedo=albedo,
                                                        specular=specular,warped_image=warped_image,world_view_transform=world_view_transform,
                                                        projection_matrix=projection_matrix,full_proj_transform=full_proj_transform,
                                                        camera_center=camera_center,intrinsic_matrix=intrinsic_matrix,K=intrinsic_matrix,
                                                        R=R,T=T,extrinsic_matrix=extrinsic_matrix, shapecode=shapecode,translation_code=translation_code,
                                                        eyelid_code=eyelid_code,fullposecode=fullposecode,expcode=expcode,
                                                        FoVx =fovx,FoVy=fovy,image_width=self.imagew,image_height=self.imageh,
                                                        image_name=imagename, timestep=original_idx))
            
    def __getitem__(self, index):
        
        cam_param = self.cam_params_list[index]
        if not self.pre_load:
            cam_param.original_image,cam_param.gt_alpha_mask,cam_param.albedo,cam_param.specular,cam_param.warped_image = \
                self._load_images(self.imagepath_list[index], self.args, self.bg)
        
        return cam_param
    
    def __len__(self, ):
        return len(self.imagepath_list)
    
            
    def _load_images(self,imagepath,args,bg):
        bg=bg.numpy()
        image = Image.open(imagepath)
        pre_path = os.path.dirname(os.path.dirname(imagepath))
        image_basename=os.path.basename(imagepath)
        mask_prepath = os.path.join(pre_path, "mask")
        
        imagew,imageh=image.size[0],image.size[1]
        image = np.array(image,dtype=np.float32)/255.0
        if image.shape[2]==4:
            mask_data=image[:,:,3:4]
            image=image[:,:,:3]
        elif os.path.exists(mask_prepath):
            maskpath=os.path.join(mask_prepath,image_basename)
            mask=Image.open(maskpath)
            mask_data=(np.array(mask,dtype=np.float32)/255.0).mean(axis=2,keepdims=True)
        else:
            mask_data=np.ones((imageh,imagew,1))
            
        image = image*mask_data+(1-mask_data)*bg
        specular,albedo=None,None
        global landmark
        if args.with_intrinsic_supervise:
            albedo_path=os.path.join(pre_path,"albedo",image_basename)
            
            if os.path.exists(albedo_path):
                albedo=Image.open(albedo_path)
                albedo=np.array(albedo,dtype=np.float32)/255.0
                albedo=albedo*mask_data+(1-mask_data)*bg
                albedo=torch.tensor(albedo, device=self.device,dtype=torch.float32).permute(2,0,1)
                
            # specular_path=os.path.join(pre_path,"specular",image_basename)
            # if os.path.exists(specular_path):
            #     specular=Image.open(specular_path)
            #     specular=(np.array(specular,dtype=np.float32)/255.0).mean(axis=2,keepdims=True)
            #     specular=specular*mask_data+(1-mask_data)*bg
            #     specular=torch.tensor(specular, device=self.device,dtype=torch.float32)
            
        warped_image=None
        crop_size=[224,224]
        if args.with_param_net_smirk:
            kpt_mediapipe = run_mediapipe((image*255.0).astype(np.uint8),self.detector)
            if (kpt_mediapipe is None):
                print('Could not find landmarks for the image using last landmark.')
                kpt_mediapipe=landmark
            else:
                kpt_mediapipe = kpt_mediapipe[..., :2]
                landmark=kpt_mediapipe
            tform = crop_face(image,kpt_mediapipe,scale=1.4,image_size=224)
            warped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True)
            warped_image=torch.tensor(warped_image, device=self.device,dtype=torch.float32)
            warped_image=warped_image.permute(2, 0, 1)[None]
            # warped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
            
        image=torch.tensor(image, device=self.device,dtype=torch.float32).permute(2, 0, 1)
        mask_data=torch.tensor(mask_data, device=self.device,dtype=torch.float32).permute(2, 0, 1)
        #adhere to Guassian splatting data reader
        image=(((image*255).to(torch.uint8))/255.0).to(torch.float32)
        mask_data=(((mask_data*255).to(torch.uint8))/255.0).to(torch.float32)
        albedo=(((albedo*255).to(torch.uint8))/255.0).to(torch.float32) if albedo is not None else None
        warped_image=(((warped_image*255).to(torch.uint8))/255.0).to(torch.float32) if warped_image is not None else None
    
        return image,mask_data,albedo,specular,warped_image
    
    def _load_camera(self,tracked_params_dict,imagekey,fovx,fovy,zfar=100.0,znear =0.01):
        
        w2c=np.array([[1 ,0 ,0 ,0 ],
                    [0 ,-1,0 ,0 ],
                    [0 ,0 ,-1,0 ],
                    [0 ,0 ,0 ,1 ]],dtype=np.float32)@np.array(tracked_params_dict[imagekey]["world_mat"],dtype=np.float32)
        R=np.transpose(w2c[:3,:3])
        T=w2c[:3, 3]
        fo=fov2focal(fovx, self.imagew)
        K= np.array([
            [fo, 0, self.imagew/2],
            [0, fo, self.imageh /2],
            [0, 0, 1]
        ],dtype=np.float32)
        
        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).to(self.device)
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1).to(self.device)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        extrinsic_matrix=world_view_transform.transpose(0,1).cuda().contiguous()# cam2world
        intrinsic_matrix=torch.tensor(K,dtype=torch.float32,device=self.device)
        R=torch.tensor(R,dtype=torch.float32,device=self.device)
        T=torch.tensor(T,dtype=torch.float32,device=self.device)
        return world_view_transform,projection_matrix,full_proj_transform,camera_center,extrinsic_matrix,intrinsic_matrix,R,T
    
    
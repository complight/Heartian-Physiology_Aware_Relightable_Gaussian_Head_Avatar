"""
Copyright (c) 2025 Eastbean Zhang. All rights reserved.

This code is a modified version of the original work from:
https://github.com/graphdeco-inria/gaussian-splatting

"""

import numpy as np
from utils.general_utils import PILtoTorch,nptoTorch
from utils.graphics_utils import fov2focal, getWorld2View2, getProjectionMatrix
import torch
import copy
from tqdm import tqdm
from copy import deepcopy
WARNED = False

def camera_to_JSON(id, camera ):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world

class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """
    @staticmethod
    def sample(horizontal_mean, vertical_mean, lookat_position,FoVx,FoVy, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cuda:0'):
        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, np.pi - 1e-5)

        theta = h
        v = v / np.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(np.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(np.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        forward_vectors = normalize_vecs(lookat_position - camera_origins)
        c2w=create_cam2world_matrix(forward_vectors, camera_origins)
        # already in COLMAP (Y down, Z forward)
        w2c=torch.tensor([[1 ,0 ,0 ,0 ],#
                    [0 ,1,0 ,0 ],
                    [0 ,0 ,1,0 ],
                    [0 ,0 ,0 ,1 ]],dtype=torch.float32,device=device)@torch.linalg.inv(c2w).squeeze(0)
        R = torch.transpose(w2c[:3,:3],0,1).cpu().numpy()
        T= w2c[:3, 3].cpu().numpy()
        zfar = 100.0
        znear = 0.01
        trans=np.array([0.0, 0.0, 0.0])
        scale=1.0
        world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(device)#w2c
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).to(device)#P:Projection matrix from cone to cube
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)#w2c@P
        camera_center = world_view_transform.inverse()[3, :3]

        return {"world_view_transform":world_view_transform,
                "projection_matrix":projection_matrix,
                "full_proj_transform":full_proj_transform,
                "camera_center":camera_center,
                "R":R,"T":T}
        
def generate_multi_view_poses(cam_params,pitch_range = 0.3,yaw_range = 0.25,num_keyframes=120,y_offset=0.0):
        #pitch_range = 0.3,yaw_range = 0.35,num_keyframes=120
        cam_params=[data for data in cam_params]
        cam_params=copy.deepcopy(cam_params)
        world2cam_views=[]
        device=cam_params[0].world_view_transform.device
        FoVx=cam_params[0].FoVx
        FoVy=cam_params[0].FoVy
        radius=cam_params[0].camera_center.square().sum().sqrt()
        lookat_position=[cam_params[0].camera_center[0].item(),cam_params[0].camera_center[1].item()+0.05,0.0]
        print("Generate multi-view poses for rendering")
        for frame_idx in tqdm(range(num_keyframes)):
            world2cam_views.append (
                LookAtPoseSampler.sample(
                horizontal_mean=3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes)),
                vertical_mean=3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes)),
                lookat_position=torch.Tensor([0.0,-y_offset,0.0]).to(device),FoVx=FoVx,FoVy=FoVy ,radius=radius, device=device)
            )
        for idx in range(len(cam_params)):
            cam_params[idx].world_view_transform=world2cam_views[idx%num_keyframes]["world_view_transform"]
            cam_params[idx].projection_matrix=world2cam_views[idx%num_keyframes]["projection_matrix"]
            cam_params[idx].full_proj_transform=world2cam_views[idx%num_keyframes]["full_proj_transform"]
            cam_params[idx].camera_center=world2cam_views[idx%num_keyframes]["camera_center"]
            cam_params[idx].R=world2cam_views[idx%num_keyframes]["R"]
            cam_params[idx].T=world2cam_views[idx%num_keyframes]["T"]
        return cam_params
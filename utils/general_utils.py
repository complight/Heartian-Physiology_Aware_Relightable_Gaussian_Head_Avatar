"""
Copyright (c) 2025 Eastbean Zhang. All rights reserved.

This code is a modified version of the original work from:
https://github.com/graphdeco-inria/gaussian-splatting

"""

import torch
import sys
from datetime import datetime
import numpy as np
import random
import re,cv2
from skimage.transform import estimate_transform
from PIL import Image
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    
def nptoTorch(image, resolution):
    resized_image = cv2.resize(image, resolution)
    resized_image=torch.from_numpy(resized_image).to(torch.float32)
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def save_image_L(image,path):
    ndarr = image.mul(255).add_(0.5).clamp_(0, 255).squeeze().to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr,mode="L")
    im.save(path, format=None)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def get_delayed_lr_func_with_decay(warmup_steps, inner_func):
    """
    warmup_steps iter lr=0 -> activate inner_func after warmup steps starting from 0 step
    """
    def helper(step):
        if step < warmup_steps:
            return 0.0
        return inner_func(step - warmup_steps)
    return helper

def get_delayed_lr_func(warmup_steps, lr_value):
    def helper(step):
        if step < warmup_steps:
            return 0.0
        return lr_value
    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def to8b(img):
    return (255 * np.clip(img, 0, 1)).astype(np.uint8)

def natural_sort_key(s):
    
    sub_strings = re.split(r'(\d+)', s)

    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    return sub_strings

def bbox2point( left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type=='kpt68':
        old_size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    elif type=='bbox':
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
    else:
        raise NotImplementedError
    return old_size, center

def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform

import mediapipe as mp
def run_mediapipe(image_numpy,detector):
    # STEP 3: Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_numpy)
    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)

    if len (detection_result.face_landmarks) == 0:
        print('No face detected')
        return None
    
    face_landmarks = detection_result.face_landmarks[0]

    face_landmarks_numpy = np.zeros((478, 3))

    for i, landmark in enumerate(face_landmarks):
        face_landmarks_numpy[i] = [landmark.x*image.width, landmark.y*image.height, landmark.z]

    return face_landmarks_numpy
def flip_align_view(normal, viewdir):
    # normal: (N, 3), viewdir: (N, 3)
    dotprod = torch.sum(
        normal * -viewdir, dim=-1, keepdims=True) # (N, 1)
    non_flip = dotprod>=0 # (N, 1)
    normal_flipped = normal*torch.where(non_flip, 1, -1) # (N, 3)
    return normal_flipped, non_flip

def get_minimum_axis(scales, rotations):
    sorted_idx = torch.argsort(scales, descending=False, dim=-1)
    R = build_rotation(rotations)
    R_sorted = torch.gather(R, dim=2, index=sorted_idx[:,None,:].repeat(1, 3, 1)).squeeze()
    x_axis = R_sorted[:,0,:] # normalized by defaut

    return x_axis

pixel_camera = None
def sample_camera_rays(HWK, R, T):
    H,W,K = HWK
    R = R.T.detach() # NOTE!!! the R rot matrix is transposed save in 3DGS
    
    global pixel_camera
    if pixel_camera is None or pixel_camera.shape[0] != H:
        K = K.astype(np.float32)
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32),
                        indexing='xy')
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_camera = torch.tensor(pixel_camera).cuda()

    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d.reshape(H,W,3)
    return rays_d
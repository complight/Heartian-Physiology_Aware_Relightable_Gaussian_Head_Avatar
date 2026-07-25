import os
import  sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from glob import glob
import json,re

# GLOBAL_POSE: if true, optimize global rotation, otherwise, only optimize head rotation (shoulder stays un-rotated)
# if GLOBAL_POSE is set to false, global translation is used.
GLOBAL_POSE = True

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils import lossfunc

import cv2
import argparse

np.random.seed(0)


def projection(points, K, w2c, no_intrinsics=False):
    rot = w2c[:, np.newaxis, :3, :3]
    points_cam = torch.sum(points[..., np.newaxis, :] * rot, -1) + w2c[:, np.newaxis, :3, 3]
    if no_intrinsics:
        return points_cam

    points_cam_projected = points_cam
    points_cam_projected[..., :2] /= points_cam[..., [2]]
    points_cam[..., [2]] *= -1

    i = points_cam_projected[..., 0] * K[0] + K[2]
    j = points_cam_projected[..., 1] * K[1] + K[3]
    points2d = torch.stack([i, j, points_cam_projected[..., -1]], dim=-1)
    return points2d


def inverse_projection(points2d, K, c2w):
    i = points2d[:, :, 0]
    j = points2d[:, :, 1]
    dirs = torch.stack([(i - K[2]) / K[0], (j - K[3]) / K[1], torch.ones_like(i) * -1], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:, np.newaxis, :3, :3], -1)
    rays_d = F.normalize(rays_d, dim=-1)
    rays_o = c2w[:, np.newaxis, :3, -1].expand(rays_d.shape)

    return rays_o, rays_d

def eyelid_close_loss(opt_lmks, target_lmks):
    upper_eyelid_lmk_ids = [ 37, 38, 43,44]
    lower_eyelid_lmk_ids = [41,40,47,46]
    diff_opt = opt_lmks[:, upper_eyelid_lmk_ids, :] - opt_lmks[:, lower_eyelid_lmk_ids, :]
    diff_target = target_lmks[:, upper_eyelid_lmk_ids, :] - target_lmks[:, lower_eyelid_lmk_ids, :]
    diff = torch.pow(diff_opt - diff_target, 2)
    return diff.mean()

class Optimizer(object):
    def __init__(self, device='cuda:0',args=None):
        deca_cfg.model.use_tex = False
        # TODO: landmark_embedding.npy with eyes to optimize iris parameters
        deca_cfg.model.flame_lmk_embedding_path = os.path.join(deca_cfg.deca_dir, 'data',
                                                               'landmark_embedding_with_eyes.npy')
        deca_cfg.rasterizer_type = 'pytorch3d' # or 'standard'
        deca_cfg.model.n_shape=args.n_shape
        deca_cfg.model.n_exp=args.n_expr
        self.deca = DECA(config=deca_cfg, device=device)
        self.with_eyelid=False
        if args is not None:
            self.with_eyelid=args.with_eyelid
            self.with_translation=args.with_translation
            self.with_translation_camera=args.with_translation_camera

    def optimize(self, shape, exp, landmark, pose, name, visualize_images, savefolder, intrinsics, json_path, size,
                 save_name,args):
        # 
        num_img = pose.shape[0]
        # we need to project to [-1, 1] instead of [0, size], hence modifying the cam_intrinsics as below
        cam_intrinsics = torch.tensor(
            [-1 * intrinsics[0] / size * 2, intrinsics[1] / size * 2, intrinsics[2] / size * 2 - 1,
             intrinsics[3] / size * 2 - 1]).float().cuda()
        
        if GLOBAL_POSE:
            translation_p = torch.tensor([0, 0, -4]).float().cuda()
        if self.with_translation_camera or not GLOBAL_POSE:
            translation_p = torch.tensor([0, 0, -4]).unsqueeze(0).expand(num_img, -1).float().cuda()
        translation_p = nn.Parameter(translation_p)
            
        if self.with_translation:
            translation=torch.zeros_like(pose[:, :3]).unsqueeze(1)
            translation=nn.Parameter(translation)
            

        if GLOBAL_POSE:
            pose = torch.cat([torch.zeros_like(pose[:, :3]), pose], dim=1)
        if landmark.shape[1] == 70:
            # use iris landmarks, optimize gaze direction
            use_iris = True
        if use_iris:
            pose = torch.cat([pose, torch.zeros_like(pose[:, :6])], dim=1)
        
        pose = nn.Parameter(pose)
        exp = nn.Parameter(exp)
        shape = nn.Parameter(shape)
        
        # set optimizer 1e-2
        lr_opt=1e-2
        if json_path is None:
            opt_p = torch.optim.Adam(
                [pose, exp, shape],
                lr=lr_opt)
        else:
            opt_p = torch.optim.Adam(
                [pose, exp],
                lr=lr_opt)
        
        if args.with_eyelid:
            eyelid=torch.zeros_like(exp[:,:2],device=exp.device)
            eyelid=nn.Parameter(eyelid)
            eyelid_params={'params': [eyelid], 'lr': 1e-3, 'name': ['eyelid']}
            opt_p.add_param_group(eyelid_params)
        else:
            eyelid=None
        if args.with_translation:
            translation_params={'params': [translation], 'lr': 1e-4, 'name': ['translation']}
            opt_p.add_param_group(translation_params)
        else:
            translation=None
        if 1:
            lr_p=1e-2
            translation_p_params={'params': [translation_p], 'lr': lr_p, 'name': ['translation_p']}
            opt_p.add_param_group(translation_p_params)
        # optimization steps
        print(shape.shape,exp.shape)
        len_landmark = landmark.shape[1]
        avg_lmk_loss=0.0
        for k in range(1,1001):
            full_pose = pose
            if not use_iris:
                full_pose = torch.cat([full_pose, torch.zeros_like(full_pose[..., :6])], dim=1)
            if not GLOBAL_POSE:
                full_pose = torch.cat([torch.zeros_like(full_pose[:, :3]), full_pose], dim=1)
            verts_p, landmarks2d_p, landmarks3d_p = self.deca.flame(shape_params=shape.expand(num_img, -1),
                                                                    expression_params=exp,
                                                                    full_pose=full_pose,eyelid_params=eyelid,
                                                                    translation_params=translation)
            # CAREFUL: FLAME head is scaled by 4 to fit unit sphere tightly
            verts_p *= 4
            landmarks3d_p *= 4
            landmarks2d_p *= 4
            # if k%300==0:
            #     lr_opt/=2
            #     for param_group in opt_p.param_groups:
            #         param_group['lr'] = lr_opt
            # perspective projection
            # Global rotation is handled in FLAME, set camera rotation matrix to identity
            ident = torch.eye(3).float().cuda().unsqueeze(0).expand(num_img, -1, -1)
            if self.with_translation_camera :
                w2c_p = torch.cat([ident, translation_p.unsqueeze(2)], dim=2)
            elif GLOBAL_POSE:
                w2c_p = torch.cat([ident, translation_p.unsqueeze(0).expand(num_img, -1).unsqueeze(2)], dim=2)
            
            else:
                w2c_p = torch.cat([ident, translation_p.unsqueeze(2)], dim=2)

            trans_landmarks2d = projection(landmarks2d_p, cam_intrinsics, w2c_p)
            ## landmark loss
            landmark_loss2 = lossfunc.l2_distance(trans_landmarks2d[:, :len_landmark, :2], landmark[:, :len_landmark])
            total_loss = landmark_loss2 + torch.mean(torch.square(shape)) * 1e-2 + torch.mean(torch.square(exp)) * 1e-2
            total_loss += torch.mean(torch.square(exp[1:] - exp[:-1])) * 1e-1
            
            total_loss += torch.mean(torch.square(pose[1:] - pose[:-1])) *args.lambda_pose_diff
            if self.with_eyelid:
                total_loss += torch.mean(torch.square(eyelid[1:] - eyelid[:-1])) * 1e-1
                # total_loss += torch.mean(torch.square(eyelid[:]-0)) * 1e-4
                eyelid_lmk_loss=eyelid_close_loss(trans_landmarks2d[:, :len_landmark, :2],landmark)
                total_loss +=eyelid_lmk_loss
            if self.with_translation:
                total_loss += torch.mean(torch.square(translation[1:] - translation[:-1])) *args.lambda_translation_diff
                
            if self.with_translation_camera:
                total_loss += torch.mean(torch.square(translation_p[1:] - translation_p[:-1])) * 5
            if not GLOBAL_POSE :
                total_loss += torch.mean(torch.square(translation_p[1:] - landmark[:, :len_landmark])) * 10

            opt_p.zero_grad()
            total_loss.backward()
            opt_p.step()
            avg_lmk_loss+=landmark_loss2.item()
            # visualize
            if k % 100 == 0:
                with torch.no_grad():
                    loss_info = '----iter: {}, time: {}\n'.format(k,
                                                                  datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                    loss_info = loss_info + f"landmark_loss: {landmark_loss2}"
                    avg_lmk_loss=0.0
                    print(loss_info)
                    if self.with_eyelid:
                        print(f"eyelid_lmk_loss:{eyelid_lmk_loss.item()}")
                    trans_verts = projection(verts_p[::50], cam_intrinsics, w2c_p[::50])
                    # trans_landmarks2d_for_visual = projection(landmarks2d_p, cam_intrinsics, w2c_p)
                    shape_images = self.deca.render.render_shape(verts_p[::50], trans_verts)
                    visdict = {
                        'inputs': visualize_images,
                        'gt_landmarks2d': util.tensor_vis_landmarks(visualize_images, landmark[::50]),
                        'landmarks2d': util.tensor_vis_landmarks(visualize_images, trans_landmarks2d.detach()[::50]),
                        'shape_images': shape_images
                    }
                    cv2.imwrite(os.path.join(savefolder, 'optimize_vis.jpg'), self.deca.visualize(visdict))

                    # shape_images = self.deca.render.render_shape(verts_p, trans_verts)
                    # print(shape_images.shape)

                    save = True
                    if save:
                        save_intrinsics = [-1 * intrinsics[0] / size, intrinsics[1] / size, intrinsics[2] / size,
                                           intrinsics[3] / size]
                        dict = {}
                        frames = []
                        for i in range(num_img):
                            save_params_dict={'file_path': './image/' + name[i],
                                           'world_mat': w2c_p[i].detach().cpu().numpy().tolist(),
                                           'expression': exp[i].detach().cpu().numpy().tolist(),
                                           
                                           'pose': full_pose[i].detach().cpu().numpy().tolist(),
                                           'bbox': torch.stack(
                                               [torch.min(landmark[i, :, 0]), torch.min(landmark[i, :, 1]),
                                                torch.max(landmark[i, :, 0]), torch.max(landmark[i, :, 1])],
                                               dim=0).detach().cpu().numpy().tolist(),
                                           'flame_keypoints': trans_landmarks2d[i, :,
                                                              :2].detach().cpu().numpy().tolist()
                                           }
                            if self.with_eyelid:
                                save_params_dict["eyelids"]=eyelid[i].detach().cpu().numpy().tolist()
                            if args.with_translation:
                                save_params_dict["translation"]=translation[i].detach().cpu().numpy().tolist()
                            frames.append(save_params_dict)

                        dict['frames'] = frames
                        dict['intrinsics'] = save_intrinsics
                        dict['shape_params'] = shape[0].cpu().numpy().tolist()
                        
                        with open(os.path.join(savefolder, save_name + '.json'), 'w') as fp:
                            json.dump(dict, fp)
        if use_iris:
            print('Optimizing iris')                
            pose = pose.detach().clone()
            translation_p = translation_p.detach().clone()
            exp = exp.detach().clone()
            shape = shape.detach().clone()
            if eyelid is not None:
                eyelid = eyelid.detach()
            if translation is not None:
                translation = translation.detach().clone()
            eye_pose = pose[:, -6:].clone()
            eye_pose = eye_pose + torch.randn_like(eye_pose) * 1e-3 # Modified to handle extreme case
            eye_pose = nn.Parameter(eye_pose, requires_grad=True)
            # set optimizer
            opt_p = torch.optim.Adam(
                [eye_pose],
                lr=1e-2)
            len_landmark = landmark.shape[1]
            avg_lmk_loss=0.0
            for k in range(1,501):
                full_pose = pose.detach().clone()
                full_pose[:, -6:] = eye_pose
                verts_p, landmarks2d_p, landmarks3d_p = self.deca.flame(shape_params=shape.expand(num_img, -1),
                                                                        expression_params=exp,
                                                                        full_pose=full_pose,eyelid_params=eyelid,
                                                                        translation_params=translation)
                # CAREFUL: FLAME head is scaled by 4 to fit unit sphere tightly
                verts_p *= 4
                landmarks3d_p *= 4
                landmarks2d_p *= 4

                # perspective projection
                # Global rotation is handled in FLAME, set camera rotation matrix to identity
                ident = torch.eye(3).float().cuda().unsqueeze(0).expand(num_img, -1, -1)
                if self.with_translation_camera :
                    w2c_p = torch.cat([ident, translation_p.unsqueeze(2)], dim=2)
                elif GLOBAL_POSE:
                    w2c_p = torch.cat([ident, translation_p.unsqueeze(0).expand(num_img, -1).unsqueeze(2)], dim=2)
                
                else:
                    w2c_p = torch.cat([ident, translation_p.unsqueeze(2)], dim=2)

                trans_landmarks2d = projection(landmarks2d_p, cam_intrinsics, w2c_p)
                ## landmark loss
                landmark_loss2 = lossfunc.l2_distance(trans_landmarks2d[:, -2:, :2], landmark[:, -2:])*10
                if k == 1:
                    print('----iter: 0, landmark_loss:', landmark_loss2.item())
                total_loss =landmark_loss2+ torch.mean(torch.square(eye_pose[1:] - eye_pose[:-1])) *args.lambda_pose_diff*0.1
    
                opt_p.zero_grad()
                total_loss.backward()
                opt_p.step()
                avg_lmk_loss+=landmark_loss2.item()
                # visualize
                if k % 100 == 0:
                    with torch.no_grad():
                        loss_info = '----iter: {}, time: {}\n'.format(k,
                                                                    datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                        loss_info = loss_info + f"landmark_loss: {landmark_loss2}"
                        avg_lmk_loss=0.0
                        print(loss_info)
            
            save_intrinsics = [-1 * intrinsics[0] / size, intrinsics[1] / size, intrinsics[2] / size,
                            intrinsics[3] / size]
            dict = {}
            frames = []
            for i in range(num_img):
                full_pose[i, -6:] = eye_pose[i]
                save_params_dict={'file_path': './image/' + name[i],
                            'world_mat': w2c_p[i].detach().cpu().numpy().tolist(),
                            'expression': exp[i].detach().cpu().numpy().tolist(),
                            
                            'pose': full_pose[i].detach().cpu().numpy().tolist(),
                            'bbox': torch.stack(
                                [torch.min(landmark[i, :, 0]), torch.min(landmark[i, :, 1]),
                                    torch.max(landmark[i, :, 0]), torch.max(landmark[i, :, 1])],
                                dim=0).detach().cpu().numpy().tolist(),
                            'flame_keypoints': trans_landmarks2d[i, :,
                                                :2].detach().cpu().numpy().tolist()
                            }
                if self.with_eyelid:
                    save_params_dict["eyelids"]=eyelid[i].detach().cpu().numpy().tolist()
                if args.with_translation:
                    save_params_dict["translation"]=translation[i].detach().cpu().numpy().tolist()
                frames.append(save_params_dict)

            dict['frames'] = frames
            dict['intrinsics'] = save_intrinsics
            dict['shape_params'] = shape[0].cpu().numpy().tolist()
            
            with open(os.path.join(savefolder, save_name + '.json'), 'w') as fp:
                json.dump(dict, fp)
        
        return dict

    def run(self, deca_code_file, face_kpts_file, iris_file, savefolder, image_path, json_path, intrinsics, size,
            save_name,n_shape,n_expr,args):
        image_path=os.path.join(savefolder,"image")
        if not os.path.exists(image_path):
            image_path=os.path.join(savefolder,"images")
        images_paths=sorted(glob(f'{image_path}/*.jpg') + glob(f'{image_path}/*.png'),key=natural_sort_key)
        deca_code = json.load(open(deca_code_file, 'r'))
        face_kpts = json.load(open(face_kpts_file, 'r'))
        try:
            iris_kpts = json.load(open(iris_file, 'r'))
        except:
            iris_kpts = None
            print("Not using Iris keypoint")
        visualize_images = []
        shape = []
        exps = []
        landmarks = []
        poses = []
        name = []
        num_img = len(deca_code)
        # ffmpeg extracted frames, index starts with 1
        for k in range(1, num_img + 1):
            image_basename=os.path.basename(images_paths[k-1])#str(k)
            shape.append(torch.tensor(deca_code[image_basename]['shape']).float().cuda())
            shape[-1]=torch.cat([shape[-1],torch.zeros(1,n_shape-shape[-1].shape[1]).cuda()],dim=1)
            exps.append(torch.tensor(deca_code[image_basename]['exp']).float().cuda())
            if n_expr>exps[-1].shape[1]:
                exps[-1]=torch.cat([exps[-1],torch.zeros(1,n_expr-exps[-1].shape[1]).cuda()],dim=1)
            poses.append(torch.tensor(deca_code[image_basename]['pose']).float().cuda())
            name.append(image_basename)
            #landmark = np.array(face_kpts['{}.png'.format(str(k))]).astype(np.float32)
            landmark = np.array(face_kpts['{}'.format(image_basename)]).astype(np.float32)
            if iris_kpts is not None:
                #iris = np.array(iris_kpts['{}.png'.format(str(k))]).astype(np.float32).reshape(2, 2)
                iris = np.array(iris_kpts['{}'.format(image_basename)]).astype(np.float32).reshape(2, 2)
                landmark = np.concatenate([landmark, iris[[1,0], :]], 0)
            landmark = landmark / size * 2 - 1
            landmarks.append(torch.tensor(landmark).float().cuda())
            if k % 50 == 1:
                # image = cv2.imread(image_path + '/{}.png'.format(str(k))).astype(np.float32) / 255.
                image = cv2.imread(image_path + '/{}'.format(image_basename)).astype(np.float32) / 255.
                image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
                visualize_images.append(torch.from_numpy(image[None, :, :, :]).cuda())

        shape = torch.cat(shape, dim=0)
        if json_path is None:
            shape = torch.mean(shape, dim=0).unsqueeze(0)
            
        else:
            shape = torch.tensor(json.load(open(json_path, 'r'))['shape_params']).float().cuda().unsqueeze(0)
            if n_shape>shape.shape[1]:
                shape=torch.cat([shape,torch.zeros(1,n_shape-shape.shape[1]).cuda()],dim=1)
        exps = torch.cat(exps, dim=0)
        
        landmarks = torch.stack(landmarks, dim=0)
        poses = torch.cat(poses, dim=0)#[(global)3, neck (0)3, (jaw)3, eyepose (0)6]
        visualize_images = torch.cat(visualize_images, dim=0)
        # optimize
        
        op_code=self.optimize(shape, exps, landmarks, poses, name, visualize_images, savefolder, intrinsics, json_path, size,
                      save_name,args)
        tracked_params={}
        tracked_params["world_mat"]=(op_code['frames'][0]["world_mat"]).copy()
        tracked_params["world_mat"].append([0.0,0.0,0.0,1.0])
        tracked_params["intrinsics"]=intrinsics#[args.fx, args.fy, args.cx, args.cy]
        tracked_params["shapecode"]=[op_code["shape_params"]]
        for k in range(1, num_img + 1):
            image_basename=name[k-1]#str(k)
            tracked_params[image_basename]={}
            # tracked_params[str(k)]["shapecode"]=
            # tracked_params[str(k)]["texcode"]=texcode.detach().cpu().numpy().tolist()
            tracked_params[image_basename]["expcode"]=[op_code['frames'][k-1]['expression']]
            #tracked_params[str(k)]["posecode"]=deca_code[str(k)]['pose']
            tracked_params[image_basename]["fullposecode"]=[op_code['frames'][k-1]['pose']]
            tracked_params[image_basename]["cam"]=deca_code[image_basename]['cam']
            # tracked_params[str(k)]["lightcode"]=lightcode.detach().cpu().numpy().tolist()
            #tracked_params[str(k)]["scale"]=4# Flame model should scale
            
            tracked_params[image_basename]["world_mat"]=(op_code['frames'][k-1]["world_mat"])
            tracked_params[image_basename]["world_mat"].append([0.0,0.0,0.0,1.0])
            #tracked_params[str(k)]["landmark"]=((landmarks[k-1].detach().cpu().numpy()+1)*2*size).tolist()
            if args.with_eyelid:
                tracked_params[image_basename]["eyelids"]=(op_code['frames'][k-1]["eyelids"])
            if args.with_translation:
                tracked_params[image_basename]["translation"]=(op_code['frames'][k-1]["translation"])
        with open(os.path.join(savefolder, "tracked_params" + '.json'), 'w') as fjson:
            json.dump(tracked_params,fjson)
        fjson.close()
        
def natural_sort_key(s):
    
    sub_strings = re.split(r'(\d+)', s)

    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    return sub_strings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='Path to images and deca and landmark jsons')
    parser.add_argument('--shape_from', type=str, default='.', help="Use shape parameter from this video if given.")
    parser.add_argument('--save_name', type=str, default='flame_params', help='Name for json')
    parser.add_argument('--fx', type=float, default=1500)
    parser.add_argument('--fy', type=float, default=1500)
    parser.add_argument('--cx', type=float, default=256)
    parser.add_argument('--cy', type=float, default=256)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--n_shape', type=int, default=300)#n_shape
    parser.add_argument('--n_expr', type=int, default=100)#n_shape
    parser.add_argument('--with_eyelid', action="store_true",default=False)#n_shape
    parser.add_argument('--with_translation', action="store_true",default=False)#n_shape
    parser.add_argument('--with_translation_camera', action="store_true",default=False)
    parser.add_argument('--lambda_translation_diff', type=float,default=5)
    parser.add_argument('--lambda_pose_diff', type=float,default=10)
    args = parser.parse_args()
    model = Optimizer(args=args)

    image_path = os.path.join(args.path, 'image')
    if args.shape_from == '.':
        args.shape_from = None
        json_path = None
    else:
        json_path = os.path.join(args.shape_from, args.save_name + '.json')
    print("Optimizing: {}".format(args.path))
    intrinsics = [args.fx, args.fy, args.cx, args.cy]
    model.run(deca_code_file=os.path.join(args.path, 'code.json'),
              face_kpts_file=os.path.join(args.path, 'keypoint.json'),
              iris_file=os.path.join(args.path, 'iris.json'), savefolder=args.path, image_path=image_path,
              json_path=json_path, intrinsics=intrinsics, size=args.size, save_name=args.save_name,
              n_shape=args.n_shape,n_expr=args.n_expr,args=args)


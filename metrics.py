"""
Copyright (c) 2025 Eastbean Zhang. All rights reserved.

This code is a modified version of the original work from:
https://github.com/graphdeco-inria/gaussian-splatting

"""

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim,l1_loss
from submodules.lpipsPyTorch import lpips,get_lpips_criterion
 
import json,logging
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    renders_list_dirs=sorted(os.listdir(renders_dir))
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    lpips_criterion=get_lpips_criterion(net_type='vgg')
    for scene_dir in model_paths:
        if 1:
            print("Scene:", scene_dir)
            logging.info(f"Scene:{scene_dir}")
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"
            test_img_paths=sorted(os.listdir(test_dir))
            for method in test_img_paths:
                print("Method:", method)
                logging.info(f"Method:{method}")
                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                maes = []
                
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips_criterion(renders[idx], gts[idx]))
                    maes.append(l1_loss(renders[idx], gts[idx]))
                    
                
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  MAE: {:>12.7f}".format(torch.tensor(maes).mean(), ".5"))
                print("")
                logging.info(f"PSNR:{torch.tensor(psnrs).mean().item()}   SSIM:{torch.tensor(ssims).mean().item()}   LPIPS:{torch.tensor(lpipss).mean().item()}    MAE:{torch.tensor(maes).mean().item()}")
                full_dict[scene_dir][method].update({   "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "SSIM": torch.tensor(ssims).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "MAE": torch.tensor(maes).mean().item()})
                per_view_dict[scene_dir][method].update({   "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "MAE": {name: lp for lp, name in zip(torch.tensor(maes).tolist(), image_names)}
                                                            })
                renders=[]
                gts=[]
                torch.cuda.empty_cache()
                
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        # except:
        #     print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)

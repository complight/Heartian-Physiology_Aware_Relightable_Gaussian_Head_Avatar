import os,sys
import scipy.io as sio
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from abc import ABC
from glob import glob
from pathlib import Path
import argparse
import cv2
from tqdm import tqdm
import face_alignment
import numpy as np
import torch, shutil
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.general_utils import natural_sort_key


def get_bbox(image, lmks, bb_scale=2.0):
    h, w, c = image.shape
    lmks = lmks.astype(np.int32)
    x_min, x_max, y_min, y_max = np.min(lmks[:, 0]), np.max(lmks[:, 0]), np.min(lmks[:, 1]), np.max(lmks[:, 1])
    x_center, y_center = int((x_max + x_min) / 2.0), int((y_max*0.2 + y_min*0.8))
    size = int(bb_scale * 2 * max(x_center - x_min, y_center - y_min))
    xb_min, xb_max, yb_min, yb_max = max(x_center - size // 2, 0), min(x_center + size // 2, w - 1), \
        max(y_center - size // 2, 0), min(y_center + size // 2, h - 1)

    yb_max = min(yb_max, h - 1)
    xb_max = min(xb_max, w - 1)
    yb_min = max(yb_min, 0)
    xb_min = max(xb_min, 0)

    if (xb_max - xb_min) % 2 != 0:
        xb_min += 1

    if (yb_max - yb_min) % 2 != 0:
        yb_min += 1

    return np.array([xb_min, xb_max, yb_min, yb_max])


def crop_image(image, x_min, y_min, x_max, y_max):
    return image[max(y_min, 0):min(y_max, image.shape[0] - 1), max(x_min, 0):min(x_max, image.shape[1] - 1), :]


def squarefiy(image, size=512):
    h, w, c = image.shape
    if w != h:
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        image = np.pad(image, [(vp, vp), (hp, hp), (0, 0)], mode='constant')

    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)


def crop_image_bbox(image, bbox):
    xb_min = bbox[0]
    xb_max = bbox[1]
    yb_min = bbox[2]
    yb_max = bbox[3]
    cropped = crop_image(image, xb_min, yb_min, xb_max, yb_max)
    return cropped

class Crop_and_matting(Dataset, ABC):
    def __init__(self, source, args):
        self.device = 'cuda:0'
        self.config = args
        self.source_dir=source
        self.name = args.name
        self.id = args.id
        if self.id == -1:
            self.source = Path(source, args.name)
        else:
            self.source = Path(source, args.name, f'p{self.name.split("subject")[-1]}_{self.id}')
        os.makedirs(self.source,exist_ok=True)
        self.initialize()
        self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=self.device)
    
    def initialize(self):
        self.image_path = Path(self.source,'image')
        image_path=self.image_path
        
        if not image_path.exists() or len(os.listdir(str(image_path))) == 0:
            # Configuration for rPPG datasets
            if self.source_dir == 'UBFC-rPPG':
                video_file=os.path.join(self.source_dir, self.name, 'vid.avi')

                if self.config.fps==-1:
                    cap = cv2.VideoCapture(video_file)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"VIDEL FPS: {fps}")
                    self.config.fps=fps
                
                if not os.path.exists(video_file):
                    logger.error(f'[ImagesDataset] Neither images nor a video was provided! Execution has stopped! {video_file}')
                    exit(1)
                image_path.mkdir(parents=True, exist_ok=True)

                VidObj = cv2.VideoCapture(video_file)
                VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
                success, frame = VidObj.read()
                frames = list()
                while success:
                    frame = np.asarray(frame)
                    frames.append(frame)
                    success, frame = VidObj.read()
                VidObj.release()

                for idx, frame_rgb in enumerate(frames):
                    cv2.imwrite(os.path.join(image_path, f"{idx:05d}.png"), frame_rgb)

            elif self.source_dir == 'MMPD':
                f = sio.loadmat(os.path.join(self.source, f'p{self.name.split("subject")[-1]}_{self.id}.mat'))
                video = f['video']
                
                if self.config.fps == -1:
                    self.config.fps = 30 
                image_path.mkdir(parents=True, exist_ok=True)

                skipped_indices = []
                for i in range(len(video)):
                    frame = video[i]
                    frame = (np.round(frame * 255)).astype('uint8')  # float 0~1 → uint8 0~255

                    if frame.max() == 0:  # data artifact exclusion
                        print(f"Frame {i} is all zero (black), skipping.")
                        skipped_indices.append(i)
                        continue

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(image_path / f'{i:05d}.png'), frame)
                
                np.save(os.path.join(self.source, 'skipped_indices.npy'), np.array(skipped_indices))
            
            elif self.source_dir == 'PURE':
                pure_img_dir = os.path.join(self.source, self.name)

                if self.config.fps == -1:
                    self.config.fps = 30

                image_path.mkdir(parents=True, exist_ok=True)
                all_png = sorted(glob(os.path.join(pure_img_dir, '*.png')))
                for i, png_path in enumerate(all_png):
                    dst = str(image_path / f'{i:05d}.png')
                    shutil.copy(png_path, dst)

        self.images = sorted(glob(f'{image_path}/*.jpg') + glob(f'{image_path}/*.png'),key=natural_sort_key)

    def process_face(self, image):
        lmks, scores, detected_faces = self.face_detector.get_landmarks_from_image(image, return_landmark_score=True, return_bboxes=True)
        if detected_faces is None:
            lmks = None
        else:
            lmks = lmks[0]
        return lmks
    
    def change_file_name(self,image_path,save_path):
        images = sorted(glob(f'{image_path}/*.jpg') + glob(f'{image_path}/*.png'),key=natural_sort_key)
        masks = sorted(glob(f'{save_path}/*.png')+ glob(f'{save_path}/*.jpg'),key=natural_sort_key)
        assert len(images)==len(masks)
        
        file_type=masks[0].split(".")[-1]
        for i in range(len(images)):
            image_name = os.path.splitext(os.path.basename(images[i]))[0]
            shutil.move(masks[i], os.path.join(save_path, image_name + "." + file_type))
            
    def robust_video_matting(self,image_path,save_path):
        #save_video_file=os.path.join(os.path.dirname(save_path),self.name+'_matted.mp4')
        import subprocess
        print(f"face masking...{image_path}")
        command = [
            'python',
            'preprocess/submodules/RobustVideoMatting/inference.py',
            '--variant', 'resnet50',
            '--checkpoint', 'preprocess/submodules/RobustVideoMatting/rvm_resnet50.pth',
            '--device', 'cuda:0',
            '--input-source', image_path,
            '--output-alpha', save_path,
            '--output-type', 'png_sequence'
        ]
        # subprocess.run(command, capture_output=True, text=True)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
        stdout, stderr = process.communicate()
        try:
            print(stdout.decode('utf-8'))
        except UnicodeDecodeError:
            print(stdout.decode('latin1'))
        
        #resnet50 mobilenetv3
        # model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50").cuda()
        # convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")
        # print("Finish face masking...")
        # convert_video(model,image_path,output_type= 'png_sequence',output_alpha=save_path)
        
        self.change_file_name(image_path,save_path)

    
    def face_parsing(self,image_path,save_path):
        import subprocess
        print("face parsing...")
        command = [
            'python',
            'preprocess/submodules/face-parsing.PyTorch/test.py',
            "--dspth",image_path,
            "--respth",save_path
        ]
        # subprocess.run(command, capture_output=True, text=True)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
        stdout, stderr = process.communicate()
        try:
            print(stdout.decode('utf-8'))
        except UnicodeDecodeError:
            print(stdout.decode('latin1'))
        print("Finish face parsing.")
    
    def merge_maks(self,image_path,mask_path,seg_path, background_path):
        # Get the list of files
        rgb_files = sorted(os.listdir(image_path))
        mask_files = sorted(os.listdir(mask_path))

        seg_files = sorted(os.listdir(seg_path))


        print("Merging mask...")
        # Process each pair of files
        for rgb_file, mask_file, seg_file in tqdm(zip(rgb_files, mask_files, seg_files)):
            # Read the images
            rgb_img = cv2.imread(os.path.join(image_path, rgb_file), cv2.IMREAD_UNCHANGED)
            mask_img = cv2.imread(os.path.join(mask_path, mask_file), cv2.IMREAD_GRAYSCALE)
            
            # Create the alpha channel
            alpha_channel = np.ones(mask_img.shape, dtype=np.float32)

            # Set the alpha channel to 0 for the clothes area in the segmentation image (value 15)
            seg_img = cv2.imread(os.path.join(seg_path, seg_file), cv2.IMREAD_GRAYSCALE)
            if self.config.mask_clothes:
                
                alpha_channel[(seg_img == 16) | (seg_img == 0)] = 0.0
            else:
                alpha_channel[ (seg_img == 0)] = 0.0
            alpha_channel = (mask_img / 255.0) * alpha_channel

            # Background Ablation: save temporal real background
            if background_path is not None:
                bg_mask = (1.0 - alpha_channel)
                bg_mask_expanded = np.expand_dims(bg_mask, axis=2)
                bg_img = (rgb_img.astype(np.float32) * bg_mask_expanded).astype(np.uint8)

                bg_alpha = (bg_mask * 255).astype(np.uint8)
                bg_rgba = cv2.merge((bg_img, bg_alpha))
                cv2.imwrite(os.path.join(background_path, rgb_file), bg_rgba)
            
            # Apply alpha to the RGB image
            alpha_expanded = np.expand_dims(alpha_channel, axis=2)
            rgb_img = (rgb_img / 255.0 * alpha_expanded) * 255
            rgb_img = rgb_img.astype(np.uint8)

            # Merge the RGB image with the alpha channel
            alpha_channel = (alpha_channel * 255).astype(np.uint8)
            rgba_img = cv2.merge((rgb_img, alpha_channel))

            # Save the result
            output_path = os.path.join(image_path, rgb_file)
            cv2.imwrite(output_path, rgba_img)

        print("Processing complete!")

    def run(self):

        
        logger.info('Croping dataset...')
        bbox = None
        for imagepath in tqdm(self.images):
            if 1:
                image = cv2.imread(imagepath)
                if self.config.crop_range is not None:
                    image = image[self.config.crop_range[0]:self.config.crop_range[1], self.config.crop_range[2]:self.config.crop_range[3],:]
                h, w, c = image.shape
                
                if bbox is None :
                    print("geting box")
                    lmk = self.process_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # estimate initial bbox
                    bbox = get_bbox(image, lmk, bb_scale=self.config.bbox_scale)
                    # torch.save(bbox, bbox_path)

                if self.config.crop_image and self.config.crop_image:
                    image = crop_image_bbox(image, bbox)
                    if self.config.image_size[0] == self.config.image_size[1]:
                        image = squarefiy(image, size=self.config.image_size[0])
                elif image.shape[0] != self.config.image_size[0] or image.shape[1] != self.config.image_size[1]:
                    image = cv2.resize(image, (self.config.image_size[1], self.config.image_size[0]), interpolation=cv2.INTER_CUBIC)
                    
                cv2.imwrite(imagepath, image)
            
        if self.config.matting:
            logger.info("Matting dataset...")
            save_mask_path=os.path.join(self.source,"mask")
            os.makedirs(save_mask_path,exist_ok=True)
            save_seg_path=os.path.join(self.source,"seg")
            os.makedirs(save_mask_path,exist_ok=True)
            self.robust_video_matting(self.image_path,save_mask_path)
            
            if self.config.save_bg == True:
                save_gb_path=os.path.join(self.source,"background")
                os.makedirs(save_gb_path, exist_ok=True)
            else:
                save_gb_path=None
            self.face_parsing(self.image_path,save_seg_path)
            self.merge_maks(self.image_path,save_mask_path,save_seg_path, save_gb_path)
            shutil.rmtree(save_mask_path)
            shutil.rmtree(save_seg_path)
        logger.info("Done!")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop and matting dataset')
    parser.add_argument('--source', type=str, required=True, help='Source directory')
    parser.add_argument('--name', type=str, required=True, help='Dataset name')
    parser.add_argument('--fps', type=int, default=-1, help='Frame per second')
    parser.add_argument('--crop_image', action='store_true', help='Crop images',default=False)
    parser.add_argument('--crop_range', type=int, nargs='+', default=None, help='Crop range')
    parser.add_argument("--image_size",type=int,nargs=2,default=[512,512],help='croped image size')
    parser.add_argument("--bbox_scale",type=float,default=2.2,help='bbox scale')#2.5
    parser.add_argument("--matting", action='store_true', help='Matting images')
    parser.add_argument("--mask_clothes",type=lambda x: x.lower() in ['true', '1'], default=True, help='remove cloth')

    parser.add_argument('--save_bg', action='store_true', help='store cut-out backgrounds')

    parser.add_argument('--id', type=int, default=-1, help='video number of MMPD dataset')

    args=parser.parse_args(sys.argv[1:])
    crop_and_matting = Crop_and_matting(args.source, args)
    crop_and_matting.run()
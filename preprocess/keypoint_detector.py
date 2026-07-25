import face_alignment
from skimage import io
import argparse
import os,torch
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', type=str,  help='.')

args = parser.parse_args()

image_path = args.path + '/image/'
if not os.path.exists(image_path):
	image_path=args.path + '/images/'
print(image_path)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
save = {}
path = args.path
print(path)

filenames=[]
for filename in tqdm(os.listdir(image_path)):
	if filename.endswith(".jpg") or filename.endswith(".png"):
		filenames.append(filename)
		image = Image.open(os.path.join(image_path, filename))
		image_tensor = torch.tensor(np.array(image))
		pred=fa.get_landmarks_from_image(image_tensor[:,:,:3])
		
		save[filename] = pred[0].tolist()

json.dump(save, open(os.path.join(path, 'keypoint.json'), 'w'))

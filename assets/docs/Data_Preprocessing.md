

<h1 align="center">Data Preprocessing</h1>

## 📌 Introduction
This document provides an overview of the data preprocessing steps required for training.  
We will cover the following key steps:
- **Frame Cropping and matting**:  Extracting frames from videos, cropping images, and parsing the foreground.
- **Albedo Extraction**: Estimating albedo information.
- **Keypoints Detection and Tracking**: Estimating keypoints and tracking FLAME parameters across frames.


### 🎞️ Frame Cropping and matting
**Note:** For the **INSTA dataset** ([Download Here](https://drive.google.com/drive/folders/1LsVvr7PPwGlyK0qiTuDVUz4ihreHJgut)), this step is **not required**, as the dataset already provides **cropped** and **matted** images.
Additionally, when comparing different methods, make sure to use the same images and masks for consistency.


#### 🛠️ Installation
1. Download the [face-parsing](https://github.com/zllrunning/face-parsing.PyTorch) pre-trained model **79999_iter.pth** from:  
   [Download Here](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view?usp=drive_open)


2. Place the downloaded file into the following directory:  
preprocess/submodules/face-parsing.PyTorch/res/cp/

3. Download the [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) pre-trained model from:
  [Download Here](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth)

4. Place the downloaded file into the following directory:  
preprocess/submodules/RobustVideoMatting

####  Running the Scripts
```shell
fps=30  # Adjust according to the actual video frame rate
resize=512  # Desired image size

python preprocess/crop_and_matting.py \
           --source $data_path_dir \
           --name $data_name \
           --fps $fps \
           --image_size $resize $resize \
           --matting \
           --crop_image \
           --mask_clothes True
```

### 🎨 Albedo Extraction

We use **[Intrinsic Anything](https://github.com/zju3dv/IntrinsicAnything)** to extract albedo.

**Note:** For videos with local lighting, albedo extraction is required (e.g., the HDTF dataset used in our paper). Videos captured under uniform lighting conditions do not require albedo extraction.


#### 🛠️ Installation

1. Download the pretrained weights for albedo extraction from [Hugging Face](https://huggingface.co/spaces/LittleFrog/IntrinsicAnything/tree/main/weights).
2. Download the files in the directory `assets/intrinsic_anything/albedo`.

The folder structure should look like this:
```shell
intrinsic_anything 
    └── albedo 
        ├── checkpoints  
        │    └── last.ckpt 
        └── configs 
            └── albedo_project.yaml
```
#### Running the Scripts
```shell
base_dir=/path/to/subject
python preprocess/submodules/IntrinsicAnything/inference.py \
 --input_dir  $base_dir/image \
 --model_dir  assets/intrinsic_anything/albedo \
 --output_dir $base_dir/albedo \
 --ddim 100 --batch_size 10 --image_interval 3
 ```


### 👁️ Facial Tracking

This facial tracking process is primarily based on [IMAvatar](https://github.com/zhengyuf/IMavatar), with some modifications. This is also the pre-tracking method used in our paper.

#### 🛠️ Installation

1. Download [deca_model.tar](https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje/view) and the `generic_model.pkl` from [FLAME2020](https://flame.is.tue.mpg.de/download.php) (rename it to `generic_model2020.pkl`).
2. Place them in the `preprocess/submodules/DECA/data` folder.

#### Running the Scripts

**1. DECA FLAME Parameter Estimation**  
Navigate to the `DECA` directory and run the `demo_reconstruct.py` script for FLAME parameter estimation:
```shell
cd preprocess/submodules/DECA
base_dir=/path/to/subject
python demos/demo_reconstruct.py \
       -i $base_dir/image \
       --savefolder $base_dir/deca \
       --saveCode True \
       --saveVis False \
       --sample_step 1 \
       --render_orig False
 ```

**2.Face Landmark Detection**
```shell
cd ../..
python keypoint_detector.py --path $base_dir
 ```

**3.Iris Segmentation with FDLite**
```shell
python iris.py --path $base_dir
```

**4.Fit FLAME Parameters**
```shell
fx=1539.67462
fy=1508.93280
cx=261.442628
cy=253.231895
resize=512
python optimize.py --path $base_dir   \
  --cx $cx --cy $cy --fx $fx --fy $fy \
  --size $resize --n_shape 100 \
  --n_expr 100  --with_translation
# Add --shape_from $another_subject_dir if sharing shape parameters
```

## 🙏 Acknowledgements

We would like to express our gratitude to the following open-source repositories and datasets that greatly contributed to this project:

- [INSTA](https://github.com/Zielon/INSTA) for providing preprocessed datasets.
- [Intrinsic Anything](https://github.com/zju3dv/IntrinsicAnything) for providing albedo extraction tools.
- [IMAvatar](https://github.com/zhengyuf/IMavatar) for providing the basis for our facial tracking method.
- [DECA](https://github.com/yfeng95/DECA): For enabling robust FLAME parameter estimation.
- [FLAME](https://flame.is.tue.mpg.de/) for supplying the FLAME model for facial parameter estimation.
- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting): For providing a video matting model, which we used for background removal.  
- [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch): For enabling semantic face parsing, which was essential in data preprocessing.  
We also thank the authors and contributors of the tools and models we used throughout this research.

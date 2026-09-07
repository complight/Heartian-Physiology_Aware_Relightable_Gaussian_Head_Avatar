<p align="center">
  <h1 align="center">Heartian: Physiology-Aware Relightable Gaussian</h1>
<p align="center">

## Setup
To begin with, we use Conda for environment management. Create and activate the required environment using:
```shell
conda env create --file environment.yml
conda activate Heartian
cd submodules
git clone https://github.com/NVlabs/nvdiffrast.git
pip install nvdiffrast
pip install diff-gaussian-rasterization_c10
pip install simple-knn
```

## Data Preprocessing

### rPPG Datasets
We use three rPPG datasets containing ground-truth rPPG signals and monocular video recordings: [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg), [PURE](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure), and [MMPD](https://github.com/McJackTang/MMPD_rPPG_dataset). Please obtain the datasets following the instructions provided by their respective publishers. Before running the preprocessing code, select the required subsets for this project and organize the datasets according to the directory structures shown below.

```shell
UBFC-rPPG/
|-- subject1/
|   |-- vid.avi
|   |-- ground_truth.txt
|-- subject2/
|   |-- vid.avi
|   |-- ground_truth.txt
|...
```

```shell
PURE/
|-- 01-01/
|   |-- 01-01/
|   |-- 01-01.json
|-- 02-01/
|   |-- 02-01/
|   |-- 02-01.json
|...
```

```shell
MMPD/
|-- subject1/
|   |p1_0/
|        |-- p1_0.mat
|   |p1_4/
|        |-- p1_4.mat
|   |p1_8/
|        |-- p1_8.mat
|   |p1_16/
|        |-- p1_16.mat
|-- subject2/
|   |p2_0/
|        |-- p2_0.mat
|   |p2_4/
|        |-- p2_4.mat
|   |p2_8/
|        |-- p2_8.mat
|   |p2_16/
|        |-- p2_16.mat
|...
```

This work is built upon [HRAvatar](https://github.com/Pixel-Talk/HRAvatar), following its data preprocessing procedure described in [Data_Preprocessing](assets/docs/Data_Preprocessing.md). In this project, monocular videos undergo standard preprocessing and facial tracking to extract per-frame FLAME parameters. The preprocessing pipeline is applied to each scene/sequence as follows.

Set the `base_dir` to the path of the corresponding subject or sequence:

```shell
base_dir=/path/to/subject/in/each/scene

# Examples:
# UBFC-rPPG/subject1
# PURE/01-01
# MMPD/subject1/p1_0
```
Preprocessing procedure for each dataset:

```shell
# Step 1 Basic Preprocessing
# UBFC-rPPG
python preprocess/crop_and_matting.py --source UBFC-rPPG --name subject1 --image_size 512 512 --matting --crop_image --mask_clothes True
# PURE
python preprocess/crop_and_matting.py --source PURE --name 01-01 --image_size 512 512 --matting --crop_image --mask_clothes True
# MMPD
python preprocess/crop_and_matting.py --source MMPD --name subject1 --id 0 --image_size 512 512 --matting --crop_image --mask_clothes True

# --save_bg Save the temporal background removed during foreground matting

# Step 2 Facial Tracking
cd preprocess/submodules/DECA
python demos/demo_reconstruct.py -i $base_dir/image --savefolder $base_dir/deca --saveCode True --saveVis False --sample_step 1 --render_orig False
cd ../..
python keypoint_detector.py --path $base_dir
python iris.py --path $base_dir
cd submodules\DECA
python -m optimize --path $base_dir --cx 256.00 --cy 256.00 --fx 1536.00 --fy 1536.00 --size 512 --n_shape 100 --n_expr 100 --with_translation

cd ../../..
```

### Environment Map
Environment map filtering is described in [Filter_Envmap](assets/docs/Filter_Envmap.md)

## Training
Training proceeds in two stages: the first stage follows the same configuration to generate the baseline avatar, while the second stage introduces and trains the rPPG modulation parameters.

```shell
# UBFC-rPPG
python train.py --source_path UBFC-rPPG/subject1 --model_path output/UBFC-rPPG/subject1_baseline --eval --epoch 15 --max_reflectance 0.8 --min_reflectance 0.04 --with_envmap_consist --expression_dirs_lr 1e-7 --pose_dirs_lr 1e-7 --shape_dirs_lr 1e-8 --position_lr_init 5e-5 --position_lr_final 5e-7
python train.py --source_path UBFC-rPPG/subject1 --model_path output/UBFC-rPPG/subject1_rppg --enable_heartbeat_albedo --eval --epoch 60 --max_reflectance 0.8 --min_reflectance 0.04 --with_envmap_consist --expression_dirs_lr 1e-7 --pose_dirs_lr 1e-7 --shape_dirs_lr 1e-8 --position_lr_init 5e-5 --position_lr_final 5e-7
python render.py --model_path output/UBFC-rPPG/subject1_rppg --enable_rppg

# PURE
python train.py --source_path PURE/01-01 --model_path output/PURE/subject01-01_baseline --eval --epoch 15 --max_reflectance 0.8 --min_reflectance 0.04 --with_envmap_consist --expression_dirs_lr 1e-7 --pose_dirs_lr 1e-7 --shape_dirs_lr 1e-8 --position_lr_init 5e-5 --position_lr_final 5e-7
python train.py --source_path PURE/01-01 --model_path output/PURE/subject01-01_rppg --enable_heartbeat_albedo --eval --epoch 60 --max_reflectance 0.8 --min_reflectance 0.04 --with_envmap_consist --expression_dirs_lr 1e-7 --pose_dirs_lr 1e-7 --shape_dirs_lr 1e-8 --position_lr_init 5e-5 --position_lr_final 5e-7
python render.py --model_path output/PURE/subject01-01_rppg --enable_rppg

# MMPD
python train.py --source_path MMPD/subject1/p1_0 --model_path output/MMPD/subject1_0_baseline --eval --epoch 15 --max_reflectance 0.8 --min_reflectance 0.04 --with_envmap_consist --expression_dirs_lr 1e-7 --pose_dirs_lr 1e-7 --shape_dirs_lr 1e-8 --position_lr_init 5e-5 --position_lr_final 5e-7
python train.py --source_path MMPD/subject1/p1_0 --model_path output/MMPD/subject1_0_rppg --enable_heartbeat_albedo --eval --epoch 60 --max_reflectance 0.8 --min_reflectance 0.04 --with_envmap_consist --expression_dirs_lr 1e-7 --pose_dirs_lr 1e-7 --shape_dirs_lr 1e-8 --position_lr_init 5e-5 --position_lr_final 5e-7
python render.py --model_path output/MMPD/subject1_0_rppg --enable_rppg
```

## Rendering
Render the full-sequence reconstruction results, organized according to their corresponding timesteps.

```shell
# UBFC
python render.py --model_path output/UBFC-rPPG/subject1_rppg --enable_rppg

# PURE
python render.py --model_path output/PURE/subject01-01_rppg --enable_rppg

# MMPD
python render.py --model_path output/MMPD/subject1_0_rppg --enable_rppg
```

Additional Rendering Options:
```shell
# by adding the arguments to render.py

--with_real_bg                                  # Render with dynamic real background for ablation study
--render_relighting                             # Perform relighting render
--with_relight_background                       # Use input environment map as background during relighting
--envmap_path assets/envmaps/children_hospital  # Filtered environment map for relighting under white, warm, cool lights and real-world scenarios
```
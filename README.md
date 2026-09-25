<p align="center">
  <h1 align="center">💓Heartian: Physiology-Aware Relightable Gaussian Head Avatar</h1>
<p align="center">

<p align="center">
  <a href="https://doi.org/10.1145/3829339.3847838"><img src="https://img.shields.io/badge/DOI-10.1145%2F3829339.3847838-brightgreen" alt="DOI"></a>
  <a href="https://github.com/complight/Heartian-Physiology_Aware_Relightable_Gaussian_Head_Avatar"><img src="https://img.shields.io/badge/Code-GitHub-black" alt="Code"></a>
  <a href="https://arxiv.org/abs/2609.28539"><img src="https://img.shields.io/badge/arXiv-Preprint-red" alt="arXiv"></a>
  <a href="https://www.kaanaksit.com/assets/pdf/FanEtAl_SigAsia2026_Heartian_physiology_aware_relightable_gaussian_head_avatar.pdf"><img src="https://img.shields.io/badge/Manuscript-PDF-blue" alt="Manuscript"></a>
  <a href="https://www.kaanaksit.com/assets/pdf/FanEtAl_SigAsia2026_Supplementary_Heartian_physiology_aware_relightable_gaussian_head_avatar.pdf"><img src="https://img.shields.io/badge/Supplementary-PDF-lightgrey" alt="Supplementary"></a>
</p>

<p align="center"><b>SIGGRAPH Asia 2026 Technical Communications</b></p>

<p align="center">
  <a href="https://merryxyfan.github.io">Xiaoyue Fan</a><sup>1</sup> ·
  <a href="https://research.adobe.com/person/jose-echevarria/">Jose Echevarria</a><sup>2</sup> ·
  <a href="https://akshayparuchuri.com/">Akshay Paruchuri</a><sup>3</sup> ·
  <a href="https://kaanaksit.com">Kaan Akşit</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> University College London &nbsp;&nbsp;
  <sup>2</sup> Adobe Research &nbsp;&nbsp;
  <sup>3</sup> Stanford University
</p>

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

This work is built upon [HRAvatar](https://github.com/Pixel-Talk/HRAvatar), following its data preprocessing procedure described in [Data_Preprocessing](https://github.com/Pixel-Talk/HRAvatar/blob/main/assets/docs/Data_Preprocessing.md). In this project, monocular videos undergo standard preprocessing and facial tracking to extract per-frame FLAME parameters. The preprocessing pipeline is applied to each scene/sequence as follows.

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
Environment map filtering is described in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/merryxyfan/Heartian-Physiology_Aware_Relightable_Gaussian_Head_Avatar/blob/main/assets/envmaps/envmap_preprocessing.ipynb)

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
Render the full-sequence reconstruction results, organized according to their corresponding timesteps. After rendering, the scene directory will contain the rendered video in `.mp4` format, as well as the reconstructed data reorganized according to the original format of each dataset. These outputs can then be directly used as input to the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) for evaluation under the same data format as the corresponding benchmark.

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


## Citation

```bibtex
@inproceedings{fan2026heartian,
              author = {Fan, Xiaoyue and Echevarria, Jose and Paruchuri, Akshay and Ak{\c{s}}it, Kaan},
              title = {{💓Heartian: Physiology-Aware Relightable Gaussian Head Avatar}},
              booktitle = {SIGGRAPH Asia 2026 Technical Communications (SA Technical Communications '26)},
              year = {2026},
              month = {December 01--04},
              publisher = {Association for Computing Machinery},
              location = {Kuala Lumpur, Malaysia},
              pages = {4},
              isbn = {979-8-4007-2841-9/2026/12},
              doi = {10.1145/3829339.3847838},
              url = {https://arxiv.org/abs/2609.28539}
              }
```
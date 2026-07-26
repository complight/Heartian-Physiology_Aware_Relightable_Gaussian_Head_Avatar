<p align="center">
  <h1 align="center">Heartian: Physiology-Aware Relightable Gaussian</h1>
<p align="center">

## 📌 Introduction

## 📂 Datasets preparation
Download the insta dataset (already with extracted mask) from [INSTA](https://github.com/Zielon/INSTA). The dataset can be accessed [here](https://keeper.mpdl.mpg.de/d/5ea4d2c300e9444a8b0b/).

The HDTF videos we used can be downloaded from [here](https://drive.google.com/drive/folders/1lJMrNuvCSCDwMsd6Pz7W3cH_jXPt_fKv?usp=sharing).

## 🛠️ Setup

#### Optimizer
The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

#### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)

#### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used VS Code)
- CUDA SDK 11 for PyTorch extensions (we used 11.7)
- C++ Compiler and CUDA SDK must be compatible

### Environment Setup
Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate HRAvatar
cd submodules
git clone https://github.com/NVlabs/nvdiffrast.git
pip install nvdiffrast
pip install diff-gaussian-rasterization_c10
pip install simple-knn
```

## 🔧 Data Preprocessing

Data preprocessing for each video includes several steps: frame extraction, foreground extraction, keypoint estimation, and face tracking.

For the INSTA dataset, we directly use the provided masks.
```shell
# example script
bash preprocess/preprocess_shell/insta/bala_preprocess.sh
```

For the HDTF dataset or custom videos, you can run the following script:
```shell
# example script
bash preprocess/preprocess_shell/HDTF/marcia_preprocess.sh
```

Use Intrinsic Anything to extract albedo as pseudo-GT.
```shell
# example script
bash preprocess/preprocess_shell/extract_albedo.sh
```

For more details on data preprocessing, refer to [Data_Preprocessing](assets/docs/Data_Preprocessing.md)

Environment map filtering is described in [Filter_Envmap](assets/docs/Filter_Envmap.md)


## 🎯 Traning

For Custom DATASET
```shell
# example script
# Note: Lower learning rates can lead to better geometry 
#       but may degrade quantitative metrics (e.g., PSNR, SSIM)
CUDA_VISIBLE_DEVICES=0  python train.py --source_path /path/to/subject \
  --model_path outputs/custom/subject  --eval  --test_set_num 500  --epochs 15 \
  --max_reflectance 0.8 --min_reflectance 0.04 --with_envmap_consist \
  --expression_dirs_lr 1e-7 --pose_dirs_lr 1e-7 --shape_dirs_lr 1e-8 \
  --position_lr_init 5e-5 --position_lr_final 5e-7
```


## 🎨 Rendering

Render the training and testing results  
(This is automatically done after training by default)
```shell
# example script
CUDA_VISIBLE_DEVICES=0 python render.py  --model_path outputs/insta/bala
```

Render others
Add arguments in render.py
```shell
--skip_test # Skip rendering self-reenactment test set results
--skip_train # Skip rendering self-reenactment training set results
--render_albedo # Render albedo component
--render_normal # Render normal component
--render_irradiance # Render irradiance component
--render_specular # Render specular component
--render_roughness  # Render roughness component
--render_reflectance # Render reflectance component
--render_depth  # Render depth map
--render_envmap # Visualize optimized environment map
--render_relighting # Perform relighting render
--with_relight_background # Use input environment map as background during relighting
--envmap_path assets/envmaps/cobblestone_street  # Filtered environment map for relighting
--render_material_editing # Render material editing results (gradually increase reflectance)
--corss_source_path  # Render cross-reenactment results (specify the processed data path of another subject)
--test_static_material_edting_idxs 100 # Apply material editing to a specific image
--test_static_relight_idxs 100  # Apply relighting to a specific image
```

### Evaluation
```shell
# example script
python metrics.py --model_path outputs/insta/bala
```
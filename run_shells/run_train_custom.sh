
#Custom Dataset
CUDA_VISIBLE_DEVICES=0  python train.py --source_path /path/to/custom/subject \
  --model_path outputs/custom/subject  --eval  --test_set_num 500  --epochs 15  \
  --max_reflectance 0.8 --min_reflectance 0.04 --with_envmap_consist \
  --expression_dirs_lr 1e-7 --pose_dirs_lr 1e-7 --shape_dirs_lr 1e-8 \
  --position_lr_init 5e-5 --position_lr_final 5e-7
  
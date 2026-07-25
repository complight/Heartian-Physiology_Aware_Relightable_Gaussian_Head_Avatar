


CUDA_VISIBLE_DEVICES=0 python preprocess/submodules/IntrinsicAnything/inference.py \
 --input_dir  /path/to/subject/image \
 --model_dir  assets/intrinsic_anything/albedo \
 --output_dir /path/to/subject/albedo \
 --ddim 100 --batch_size 10 --image_interval 3



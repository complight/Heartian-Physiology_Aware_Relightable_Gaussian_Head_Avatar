# python preprocess/submodules/IntrinsicAnything/inference.py \
#  --input_dir  preprocess/submodules/IntrinsicAnything/examples  \
#  --model_dir  /mnt1/zdb/code/intrinsic_anything/albedo \
#  --output_dir preprocess/submodules/IntrinsicAnything/out/albedo \
#  --ddim 100 \
#  --batch_size 4

python preprocess/submodules/IntrinsicAnything/inference.py \
 --input_dir  /mnt2/zdb/dataset/mono_head/HDTF/haaland/image \
 --model_dir  /mnt1/zdb/code/intrinsic_anything/albedo \
 --output_dir /mnt2/zdb/dataset/mono_head/HDTF/haaland/albedo \
 --ddim 100 --batch_size 10 --image_interval 5

python preprocess/submodules/IntrinsicAnything/inference.py \
 --input_dir  /mnt2/zdb/dataset/mono_head/HDTF/haaland/image \
 --model_dir  /mnt1/zdb/code/intrinsic_anything/specular \
 --output_dir /mnt2/zdb/dataset/mono_head/HDTF/haaland/specular \
 --ddim 100 --batch_size 10 --image_interval 3
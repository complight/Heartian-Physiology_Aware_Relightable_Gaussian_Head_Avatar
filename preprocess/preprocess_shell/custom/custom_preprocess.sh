fx=1536.00
fy=1536.00
cx=256.00
cy=256.00
resize=512

data_names=("subject8")
shape_data_name="subject8"
data_path_dir="D:/CompLightLab/HRAvatar/UBFC-rPPG"

pwd="D:/CompLightLab/HRAvatar/preprocess"
path_deca=$pwd'/submodules/DECA'
visible_device=0
fps=30

echo "croping and matting video"
for data_name in "${data_names[@]}"
do
    echo $data_name
    CUDA_VISIBLE_DEVICES=$visible_device python preprocess/crop_and_matting.py --source $data_path_dir --name $data_name --fps $fps \
        --image_size $resize $resize --matting --crop_image --mask_clothes True
done

echo "DECA FLAME parameter estimation"
cd $path_deca
for data_name in "${data_names[@]}"
do
  echo $data_name
  CUDA_VISIBLE_DEVICES=$visible_device python demos/demo_reconstruct.py -i $data_path_dir/$data_name/image \
  --savefolder $data_path_dir/$data_name/"deca" --saveCode True --saveVis False --sample_step 1  --render_orig False
done

echo "face alignment landmark detector"
cd $pwd
for data_name in "${data_names[@]}"
do
  echo $data_name
  CUDA_VISIBLE_DEVICES=$visible_device python keypoint_detector.py --path $data_path_dir/$data_name
done


echo "iris segmentation with fdlite"
cd $pwd
for data_name in "${data_names[@]}"
do
  echo $data_name
  echo --path $data_path_dir/$data_name
  CUDA_VISIBLE_DEVICES=$visible_device python iris.py --path $data_path_dir/$data_name
done

echo "fit FLAME parameter for one video: "$shape_data_name
cd $path_deca
CUDA_VISIBLE_DEVICES=$visible_device python -m optimize --path $data_path_dir/$shape_data_name --cx $cx --cy $cy --fx $fx --fy $fy --size $resize \
  --n_shape 100 --n_expr 100  --with_translation 

echo "fit FLAME parameter for other videos, while keeping shape parameter fixed"
cd $path_deca
for data_name in "${data_names[@]}"
do
  if [ "$shape_data_name" == "$data_name" ];
  then
    continue
  fi
  shape_from=$data_path_dir/$shape_data_name
  echo $data_name
  CUDA_VISIBLE_DEVICES=$visible_device python optimize.py --path $data_path_dir/$data_name --shape_from $shape_from  --cx $cx --cy $cy --fx $fx --fy $fy --size $resize\
   --n_shape 100 --n_expr 100  --with_translation 
done

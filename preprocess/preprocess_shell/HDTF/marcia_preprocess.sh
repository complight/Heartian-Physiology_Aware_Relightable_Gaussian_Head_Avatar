fx=1539.67462
fy=1508.93280
cx=261.442628
cy=253.231895
resize=512

data_names=("marcia")
shape_data_name="marcia"
data_path_dir="/path/to/HDTF"
pwd="/path/to/HRAvatar/preprocess"

fps=30
path_deca=$pwd'/submodules/DECA'
visible_device=0

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
CUDA_VISIBLE_DEVICES=$visible_device python optimize.py --path $data_path_dir/$shape_data_name --cx $cx --cy $cy --fx $fx --fy $fy --size $resize \
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

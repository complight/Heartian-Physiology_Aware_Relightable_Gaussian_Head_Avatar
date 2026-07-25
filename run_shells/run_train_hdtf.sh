#HDTF Dataset (preprocessed)
# elijah haaland  katie marcia randpaul schako tom veronica

#training
for subject in elijah haaland  katie marcia randpaul schako tom veronica
do
  CUDA_VISIBLE_DEVICES=0 python train.py --source_path path/to/HDTF/$subject \
  --model_path outputs/HDTF/$subject --eval  --test_set_num 500  --epochs 15 \
  --max_reflectance 0.8 --min_reflectance 0.04 --with_envmap_consist
done


#rendering
for subject in elijah haaland  katie marcia randpaul schako tom veronica
do
  CUDA_VISIBLE_DEVICES=0 python render.py \
  --model_path outputs/HDTF/$subject \
  --skip_train --render_albedo --render_normal --render_irradiance --render_specular --render_roughness --render_reflectance \
  --render_envmap --envmap_path assets/envmaps/cobblestone_street --render_relighting --render_material_editing \
  --test_static_material_edting_idxs 150 \
  --test_static_relight_idxs 150 --with_relight_background
done



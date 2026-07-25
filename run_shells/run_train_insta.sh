#INSTA Dataset (preprocessed)
# malte_1  nf_01  nf_03 wojtek_1 obama biden person_0004 bala justin marcel

#training
for subject in wojtek_1 malte_1 nf_01 nf_03  obama biden person_0004 bala justin marcel
do
  CUDA_VISIBLE_DEVICES=1 python train.py --source_path path/to/insta/$subject \
  --model_path outputs/insta/$subject --eval --test_set_num 350 --epochs 15 --min_reflectance 0.04
done

#rendering
for subject in wojtek_1 malte_1 nf_01 nf_03  obama biden person_0004 bala justin marcel
do
  CUDA_VISIBLE_DEVICES=1 python render.py \
  --model_path outputs/insta/$subject \
  --skip_train --render_albedo --render_normal --render_irradiance --render_specular --render_roughness --render_reflectance \
  --render_envmap \
  --envmap_path assets/envmaps/cobblestone_street --render_relighting --render_material_editing \
  --test_static_material_edting_idxs 150 \
  --test_static_relight_idxs 150 --with_relight_background
done



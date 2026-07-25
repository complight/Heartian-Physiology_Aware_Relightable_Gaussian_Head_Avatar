
CUDA_VISIBLE_DEVICES=0  python render.py  \
 --model_path outputs/insta/subject \
 --skip_train --render_albedo --render_normal --render_irradiance --render_specular --render_roughness --render_reflectance \
  --render_envmap --render_depth --render_depth_normal \
 --envmap_path assets/envmaps/cobblestone_street --render_relighting --render_material_editing \
 --corss_source_path /path/to/other_subject --test_static_material_edting_idxs 150 \
 --test_static_relight_idxs 150 --with_relight_background

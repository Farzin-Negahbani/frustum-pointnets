#/bin/bash


python train/extract_mistakes.py --gpu 0 --frustum_map kitti/frustum_carpedcyc__frustum_mapping.pickle --model frustum_pointnets_v1 --model_file  train/log_FPN_V1_normal_pseudo_sparsed/model.ckpt --num_point 1024 --frustum_folder frustums_1F_normal_merged_sparse --log_dir train/log_FPN_V1_normal_pseudo_sparsed/mistakes/
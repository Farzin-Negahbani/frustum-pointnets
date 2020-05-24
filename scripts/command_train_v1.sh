#/bin/bash

#python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir train/log_FPN_V1_normal_pseudo_sparsed --num_point 1024 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --frustum_folder frustums_1F_normal_merged_sparse
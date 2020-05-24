#/bin/bash

## one frustum 
#echo "Start predictions..."
#python train/test.py --gpu 0 --num_point 1024 --model frustum_pointnets_v1 --model_path train/log_v1_1F/model.ckpt --output train/detection_results_v1_1F --data_path kitti/frustum_carpedcyc_val_rgb_detection.pickle 
#echo "Detection results are ready. Evaluating..."
#train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ train/detection_results_v1_1F

## Two Frustum 
echo "Start predictions..."
#python train/test.py --gpu 0 --num_point 1024 --model frustum_pointnets_v1 --model_path train/log_FPN_V1_normal_pseudo_sparsed/model.ckpt --output train/detection_results_1F_merged_sparsed --data_path kitti/frustums_1F_normal_merged_sparse/frustum_carpedcyc_val.pickle
echo "Detection results are ready. Evaluating..."
train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ train/detection_results_1F_merged_sparsed

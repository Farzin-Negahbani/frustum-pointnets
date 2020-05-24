#/bin/bash
# Options
#   Frutums:
#      One Frustum
#      intersection of two Frustums                             --two_frustum    
#   Merging with pseudo lidar:
#      To merge with pseudo lidars                             --merge_pseudo_lidar
#      To merge intersection of two frustums in pseudo lidar   --pseudo_two_frustum
#   generate train or validation frustums
#      validation                                              --gen_val
#      Train                                                   --gen_train
#       
#   Note: generate from RGB detections needs 2D detections for both left and right images and not implemented yet



# One Frustum 
python kitti/prepare_data.py --gen_val 

# Two Frustum
#python kitti/prepare_data.py --two_frustum  --gen_train --gen_val  --merge_pseudo_lidar  --pseudo_two_frustum
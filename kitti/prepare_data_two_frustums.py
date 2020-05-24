
'''
Prepare KITTI data for 3D object detection.

Author: Charles R. Qi
Date: September 2017

This version written for loading 
two files.
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
import kitti_util as utils
import cPickle as pickle
from kitti_object import *
import argparse


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4,2))
    box2d_corners[0,:] = [box2d[0],box2d[1]] 
    box2d_corners[1,:] = [box2d[2],box2d[1]] 
    box2d_corners[2,:] = [box2d[2],box2d[3]] 
    box2d_corners[3,:] = [box2d[0],box2d[3]] 
    box2d_roi_inds = in_hull(pc[:,0:2], box2d_corners)
    return pc[box2d_roi_inds,:], box2d_roi_inds
     
def demo():
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'))
    data_idx = 38

    # Load data from dataset
    objects = dataset.get_label_objects(data_idx)
    objects[0].print_object()
    img = dataset.get_image(data_idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_height, img_width, img_channel = img.shape
    print(('Image Left shape: ', img.shape))

    img_r = dataset.get_image_r(data_idx)
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB) 
    img_height_r, img_width_r, img_channel_r = img_r.shape
    print(('Image Right shape: ', img_r.shape))

    pc_velo = dataset.get_lidar(data_idx)[:,0:3]
    calib = dataset.get_calibration(data_idx)

    ## Draw lidar in rect camera coord
    #print(' -------- LiDAR points in rect camera coordination --------')
    #pc_rect = calib.project_velo_to_rect(pc_velo)
    #fig = draw_lidar_simple(pc_rect)
    #raw_input()

    # Draw 2d and 3d boxes on image
    print(' -------- 2D/3D bounding boxes in images --------')
    show_image_with_boxes(img, objects, calib)
    raw_input()

    # Show all LiDAR points. Draw 3d box in LiDAR point cloud
    print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')
    #show_lidar_with_boxes(pc_velo, objects, calib)
    #raw_input()
    show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
    raw_input()

    ## Visualize LiDAR points on images
    #print(' -------- LiDAR points projected to image plane --------')
    #show_lidar_on_image(pc_velo, img, calib, img_width, img_height) 
    #raw_input()
    
    # Show LiDAR points that are in the 3d box
    print(' -------- LiDAR points in a 3D bounding box --------')
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P) 
    box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, box3d_pts_3d_velo)
    print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(box3droi_pc_velo, fig=fig)
    draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
    mlab.show(1)
    raw_input()
    
    # UVDepth Image and its backprojection to point clouds
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    cameraUVDepth = np.zeros_like(imgfov_pc_rect)
    cameraUVDepth[:,0:2] = imgfov_pts_2d
    cameraUVDepth[:,2] = imgfov_pc_rect[:,2]

    # Show that the points are exactly the same
    backprojected_pc_velo = calib.project_image_to_velo(cameraUVDepth)
    print(imgfov_pc_velo[0:20])
    print(backprojected_pc_velo[0:20])

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(backprojected_pc_velo, fig=fig)
    raw_input()

    # Only display those points that fall into the left 2d box
    print(' -------- LiDAR points in the left frustum from a 2D box --------')
    xmin,ymin,xmax,ymax = \
        objects[0].xmin, objects[0].ymin, objects[0].xmax, objects[0].ymax
    boxfov_pc_velo = \
        get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax)
    print(('Left 2D box FOV point num: ', boxfov_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(boxfov_pc_velo, fig=fig)
    mlab.show(1)
    raw_input()

  
    # Only display those points that fall into the right 2d box
    print(' -------- LiDAR points in the right frustums from a 2D box Method 1--------')
    
    # We need to project xmin,ymin,xmax,ymax to the right image cordinate
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P3)
    xcord   = box3d_pts_2d[:,0]
    ycord   = box3d_pts_2d[:,1]
    xmin_r  = int(min(xcord))
    ymin_r  = int(min(ycord))
    xmax_r  = int(max(xcord))
    ymax_r  = int(max(ycord))
    box2d_r = xmin_r,ymin_r,xmax_r,ymax_r

    boxfov_pc_velo = get_lidar_in_image_fov_right(pc_velo, calib, xmin_r, ymin_r, xmax_r, ymax_r)
    
    print(('Right 2D box FOV point num Method 1: ', boxfov_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(boxfov_pc_velo, fig=fig)
    mlab.show(1)
    raw_input()

    # Draw 2d box in right image
    print(' -------- 2D bounding box in right images using Method 1 --------')
    draw_2d_box(img_r, [xmin_r, ymin_r, xmax_r, ymax_r], calib)
    raw_input()
    
    
    # Only display those points that fall into the right 2d box
    print(' -------- LiDAR points in the right frustums from a 2D box Method 2 --------')
    
    xmin_r,ymin_r,xmax_r,ymax_r = objects[0].xmin, objects[0].ymin, objects[0].xmax, objects[0].ymax

    point = np.zeros((1,3)) 
    point[0,0]= xmin_r
    point[0,1]= ymin_r
    point[0,2]= 30
    point_r = calib.project_rect_to_image_right(calib.project_image_to_rect(point))
    xmin_r = point_r[0,0]
    ymin_r = point_r[0,1]
    
    point[0,0] = xmax_r
    point[0,1] = ymax_r
    point_r = calib.project_rect_to_image_right(calib.project_image_to_rect(point))
    xmax_r = point_r[0,0]
    ymax_r = point_r[0,1] 

    boxfov_pc_velo = get_lidar_in_image_fov_right(pc_velo, calib, xmin_r, ymin_r, xmax_r, ymax_r)


    print(('Right 2D box FOV point num Method 2: ', boxfov_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(boxfov_pc_velo, fig=fig)
    mlab.show(1)
    raw_input()

    # Draw 2d box in right image
    print(' -------- 2D bounding box in right images using Method 2 --------')
    draw_2d_box(img_r, [xmin_r, ymin_r, xmax_r, ymax_r], calib)
    raw_input()
    

    # Displays those points that fall into the intersection of right and left
    print(' -------- LiDAR points in the intersection right and left frustums --------')
    xmin,ymin,xmax,ymax = objects[0].xmin, objects[0].ymin, objects[0].xmax, objects[0].ymax

    # We need to project xmin,ymin,xmax,ymax to the right image cordinate
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P3)
    xcord   = box3d_pts_2d[:,0]
    ycord   = box3d_pts_2d[:,1]
    xmin_r  = int(min(xcord))
    ymin_r  = int(min(ycord))
    xmax_r  = int(max(xcord))
    ymax_r  = int(max(ycord))
    #box2d_r = xmin_r, ymin_r, xmax_r, ymax_r

    boxfov_pc_velo = get_lidar_in_image_fov_intersect(pc_velo, calib, xmin, ymin, xmax, ymax, xmin_r, ymin_r, xmax_r, ymax_r)
    
    print((' Intersection of frustums point num: ', boxfov_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(boxfov_pc_velo, fig=fig)
    mlab.show(1)
    raw_input()

def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height 
    '''
    r = shift_ratio
    xmin,ymin,xmax,ymax = box2d
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)
    h2 = h*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    w2 = w*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])
 
"""
idx_filename   :os.path.join(BASE_DIR, 'image_sets/train.txt'),
split          :'training',
output_filename:os.path.join(BASE_DIR, output_prefix+'train.pickle'),
viz            :False,
perturb_box2d  :True,
augmentX       :5,
type_whitelist :type_whitelist,
two_frustum    :args.two_frustum


"""

def extract_frustum_data(idx_filename, split, output_filename, viz=False,
                       perturb_box2d=False, augmentX=1, type_whitelist=['Car']
                       , two_frustum=False, merge_pseudo_lidar=False,merge_th=20,
                        far_obj_th=15, pseudo_two_frustum=False, gen_mapping= False, map_outfile='frustum_mapping.pickle'):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)
        
    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
        merge_pseudo_lidar: bool, determines whether used pseudo lidar when a frustum has few points
        merge_th: scalar, Threshold to merge pseudo_lidar and lidar for frustums
        pseudo_two_frustum: in case of merge_pseudo_lidar=True, if true uses intersection of two frustums.

    Output:
        None (will write a .pickle file to the disk)
    '''

    print("Two frustum setting is: ", two_frustum)


    dataset = kitti_object(os.path.join(ROOT_DIR,'dataset/KITTI/object'), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list      = [] # int number
    box2d_list   = [] # [xmin,ymin,xmax,ymax]
    box3d_list   = [] # (8,3) array in rect camera coord
    input_list   = [] # channel number = 4, xyz,intensity in rect camera coord
    label_list   = [] # 1 for roi object, 0 for clutter
    type_list    = [] # string e.g. Car
    heading_list = [] # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = [] # array of l,w,h
    frustum_angle_list = [] # angle of 2d box center from pos x-axis

    pos_cnt     = 0 #?
    all_cnt     = 0 #?
    rej_cnt     = 0 # Reject too far away object or object without points
    acc_cnt     = 0 #?
    wo_label    = 0
    too_far_cnt = 0
    low_num_point_obj = 0 
    merge_cnt         = 0
    num_pnt_merged    = 0

    n_box3d_pts_2d_rej = 0 #Object is behind the camera
    n_box3d_pts_2d_acc = 0 #Object is not behind the camera

    # Save frustum mapping
    frustum_map={}
    frustum_id = 0


    for data_idx in data_idx_list:  #For every image
        print('------------- ', data_idx, ' ', (data_idx * 100) / 7479 ,'% ', (rej_cnt * 100) / (rej_cnt + acc_cnt + 1) , '%' )
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        
        pc_velo        = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]

        if merge_pseudo_lidar:
            pseudo_lidar   = dataset.get_pseudo_lidar(data_idx)
            pseudo_pc_rect = np.zeros_like(pseudo_lidar)
            pseudo_pc_rect[:,0:3] = calib.project_velo_to_rect(pseudo_lidar[:,0:3])
            pseudo_pc_rect[:,3] = pseudo_lidar[:,3]

        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],
            calib, 0, 0, img_width, img_height, True)

        ## CHECK img_width and img_height need to change to right one
        if two_frustum:
            _r, pc_image_coord_r, img_fov_inds_r = get_lidar_in_image_fov_right(pc_velo[:,0:3],
                calib, 0, 0, img_width, img_height, True)
        
        if merge_pseudo_lidar:
            _r, pseudo_pc_image_coord, pseudo_img_fov_inds = get_lidar_in_image_fov_right(pseudo_lidar[:,0:3],
                calib, 0, 0, img_width, img_height, True)

            _r, pseudo_pc_image_coord_r, pseudo_img_fov_inds_r = get_lidar_in_image_fov_right(pseudo_lidar[:,0:3],
                calib, 0, 0, img_width, img_height, True)
        
        for obj_idx in range(len(objects)): #For every detected object
            if objects[obj_idx].type not in type_whitelist :continue
            is_valid = True #initially right frustum is valid
            # 2D BOX: Get pts rect backprojected 
            box2d   = objects[obj_idx].box2d

            if two_frustum:
                # Calculate the right bounding box
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[obj_idx], calib.P3)

                #If object cannot be seen from the lidar box3d_pts_2d will be none.
                if not box3d_pts_2d is None:  
                    xcord   = box3d_pts_2d[:,0]
                    ycord   = box3d_pts_2d[:,1]
                    xmin_r  = int(min(xcord))
                    ymin_r  = int(min(ycord))
                    xmax_r  = int(max(xcord))
                    ymax_r  = int(max(ycord))
                    box2d_r = [xmin_r,ymin_r,xmax_r,ymax_r]
                    n_box3d_pts_2d_acc += 1
                else:
                    print ("Object behind camera, data_idx is: ", data_idx, " obj_idx is: ", obj_idx)
                    n_box3d_pts_2d_rej += 1
                    is_valid = False

            for _ in range(augmentX):
                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin,ymin,xmax,ymax         = random_shift_box2d(box2d)
                    #print("Left Box:",box2d)
                    #print(xmin,ymin,xmax,ymax)

                    if two_frustum and is_valid:
                        xmin_r,ymin_r,xmax_r,ymax_r =  random_shift_box2d(box2d_r)
                        #print("Right Box:", box2d_r)
                        #print(xmin_r,ymin_r,xmax_r,ymax_r)
                else:
                    xmin,ymin,xmax,ymax = box2d
                    if two_frustum and is_valid:
                        xmin_r, ymin_r, xmax_r, ymax_r = box2d_r
               
                #  xmin,ymin,xmax,ymax = box2d in  *rect camera* coord system
                
                box_fov_inds = (pc_image_coord[:,0]<xmax) & \
                    (pc_image_coord[:,0]>=xmin) & \
                    (pc_image_coord[:,1]<ymax) & \
                    (pc_image_coord[:,1]>=ymin)
            
                if two_frustum and is_valid:
                    box_fo_inds_r = (pc_image_coord_r[:,0]<xmax_r) & \
                        (pc_image_coord_r[:,0]>=xmin_r) & \
                        (pc_image_coord_r[:,1]<ymax_r) & \
                        (pc_image_coord_r[:,1]>=ymin_r)

                    box_fov_inds = box_fov_inds & img_fov_inds & box_fo_inds_r
                    pc_in_box_fov = pc_rect[box_fov_inds,:]

                else:
                    box_fov_inds = box_fov_inds & img_fov_inds 
                    pc_in_box_fov = pc_rect[box_fov_inds,:]
                

                if merge_pseudo_lidar:
                    '''
                    Start Merging if number of points is lower than a threshold
                    '''
                    pc_in_box_fov = pc_rect[box_fov_inds,:]

                    if  len(pc_in_box_fov)< merge_th:
                        low_num_point_obj += 1

                        ps_box_fov_inds = (pseudo_pc_image_coord[:,0]<xmax) & \
                            (pseudo_pc_image_coord[:,0]>=xmin) & \
                            (pseudo_pc_image_coord[:,1]<ymax) & \
                            (pseudo_pc_image_coord[:,1]>=ymin)
                        
                        if pseudo_two_frustum:
                            ps_box_fo_inds_r = (pseudo_pc_image_coord_r[:,0]<xmax_r) & \
                                (pseudo_pc_image_coord_r[:,0]>=xmin_r) & \
                                (pseudo_pc_image_coord_r[:,1]<ymax_r) & \
                                (pseudo_pc_image_coord_r[:,1]>=ymin_r)
        
                            ps_box_fov_inds    = ps_box_fov_inds & pseudo_img_fov_inds & ps_box_fo_inds_r
                        else:
                            ps_box_fov_inds    = ps_box_fov_inds & pseudo_img_fov_inds 

                        ps_pc_in_box_fov   = pseudo_pc_rect[ps_box_fov_inds,:]

                        if len(ps_pc_in_box_fov)>1:
                            merge_cnt +=1
                            num_pnt_merged += len(ps_pc_in_box_fov)
                            pc_in_box_fov  = np.concatenate((pc_in_box_fov,ps_pc_in_box_fov), axis=0)
                    '''
                    End
                    '''

                # Get frustum angle (according to center pixel in 2D BOX)
                box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
                uvdepth = np.zeros((1,3))
                uvdepth[0,0:2] = box2d_center
                uvdepth[0,2] = 20 # some random depth
                box2d_center_rect = calib.project_image_to_rect(uvdepth)
                frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],
                    box2d_center_rect[0,0])

                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P) 
                _,inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                label = np.zeros((pc_in_box_fov.shape[0]))
                label[inds] = 1
                # Get 3D BOX heading
                heading_angle = obj.ry
                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])
                
                # Reject too far away object or object without points
                if ymax-ymin<far_obj_th or np.sum(label)==0:
                    if ymax-ymin<far_obj_th:
                        too_far_cnt +=1
                    if np.sum(label)==0:
                        wo_label    +=1
                    rej_cnt +=1
                    continue
                
                id_list.append(data_idx)
                box2d_list.append(np.array([xmin,ymin,xmax,ymax]))
                box3d_list.append(box3d_pts_3d)
                input_list.append(pc_in_box_fov)
                label_list.append(label)
                type_list.append(objects[obj_idx].type)
                heading_list.append(heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle)

                # Save frustum mapping
                frustum_map[frustum_id] = {'img_indx':data_idx , 'obj_indx': obj_idx}
                frustum_id   += 1
                
                # collect statistics
                acc_cnt += 1
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]

    print('Average pos ratio: %f' % (pos_cnt/float(all_cnt)))    #Farzin this naming is too ambigious #Onur This line is not written by us
    print('Average npoints: %f' % (float(all_cnt)/len(id_list)))
    print('Too Far Frustums: %d' % too_far_cnt)
    print('Without Label Frustums: %d' % wo_label)
    print('Rejected Frustums: %d' % rej_cnt)
    print('Accepted Frustums: %d' % acc_cnt)
    print('Acceptance Ratio : %f' % (float(acc_cnt)/(acc_cnt+rej_cnt)))
    if merge_pseudo_lidar:
        print('Objects with less than %d points: %d' % (merge_th,low_num_point_obj))
        print('Number of Points merged from Pseudo-Lidar : %d' % num_pnt_merged)
        print('Number of Merge actions: %d' % merge_cnt)
    
    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list,fp)
        pickle.dump(box3d_list,fp)
        pickle.dump(input_list, fp)
        pickle.dump(label_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(heading_list, fp)
        pickle.dump(box3d_size_list, fp)
        pickle.dump(frustum_angle_list, fp)

    # Dumping mapping List
    if  gen_mapping:
        with open(map_outfile,'wb') as fp:
            pickle.dump(frustum_map, fp)
           
    
    if viz:
        import mayavi.mlab as mlab
        for i in range(10):
            p1 = input_list[i]
            seg = label_list[i] 
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,0], p1[:,1], p1[:,2], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()

def get_box3d_dim_statistics(idx_filename):
    ''' Collect and dump 3D bounding box statistics '''
    dataset = kitti_object(os.path.join(ROOT_DIR,'dataset/KITTI/object'))
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type=='DontCare':continue
            dimension_list.append(np.array([obj.l,obj.w,obj.h])) 
            type_list.append(obj.type) 
            ry_list.append(obj.ry)

    with open('box3d_dimensions.pickle','wb') as fp:
        pickle.dump(type_list, fp)
        pickle.dump(dimension_list, fp)
        pickle.dump(ry_list, fp)

def read_det_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3,7)]))
    return id_list, type_list, box2d_list, prob_list

 
def extract_frustum_data_rgb_detection(det_filename, split, output_filename,
                                       viz=False,
                                       type_whitelist=['Car'],
                                       img_height_threshold=25,
                                       lidar_point_threshold=5, two_frustum=False):
    ### DO NOT USE TWO FRUSTUMS

    print("Two frustum setting is: ", two_frustum)

    ''' Extract point clouds in frustums extruded from 2D detection boxes.
        Update: Lidar points and 3d boxes are in *rect camera* coord system
            (as that in 3d box label files)
        
    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        type_whitelist: a list of strings, object types we are interested in.
        img_height_threshold: int, neglect image with height lower than that.
        lidar_point_threshold: int, neglect frustum with too few points.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'), split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    cache_id = -1
    cache = None

    id_list             = []
    type_list           = []
    box2d_list          = []
    prob_list           = []
    input_list          = [] # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list  = [] # angle of 2d box center from pos x-axis

    rej = 0
    acc = 0

    for det_idx in range(len(det_id_list)):

        is_valid = True #if right frustum is valid
        data_idx = det_id_list[det_idx]
        print('det idx: %d/%d, data idx: %d' % (det_idx, len(det_id_list), data_idx))

        if cache_id != data_idx:
            calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
            pc_velo = dataset.get_lidar(data_idx)
            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
            pc_rect[:,3] = pc_velo[:,3]
            img = dataset.get_image(data_idx)
            img_height, img_width, img_channel = img.shape
            _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(\
                pc_velo[:,0:3], calib, 0, 0, img_width, img_height, True)

            if two_frustum:
                img_r = dataset.get_image(data_idx)
                img_height_r, img_width_r, img_channel_r = img_r.shape
                _r, pc_image_coord_r, img_fov_inds_r = get_lidar_in_image_fov_right(pc_velo[:,0:3],
                    calib, 0, 0, img_width_r, img_height_r, True)

            cache = [calib,pc_rect,pc_image_coord,img_fov_inds]
            cache_id = data_idx
        else:
            calib,pc_rect,pc_image_coord,img_fov_inds = cache

        if det_type_list[det_idx] not in type_whitelist: continue

        if two_frustum:
                # Calculate the right bounding box
                #project_to_image
                xmin_r,ymin_r,xmax_r,ymax_r = det_box2d_list[det_idx]

                point = np.zeros((1,3)) 
                point[0,0]= xmin_r
                point[0,1]= ymin_r
                point[0,2]= 10
                point_r = calib.project_rect_to_image_right(calib.project_image_to_rect(point))
                xmin_r = point_r[0,0]
                ymin_r = point_r[0,1]
                
                point[0,0] = xmax_r
                point[0,1] = ymax_r
                point_r = calib.project_rect_to_image_right(calib.project_image_to_rect(point))
                xmax_r = point_r[0,0]
                ymax_r = point_r[0,1] 

                ## Fixing nonetype projections for some 2D boxes should take a furthur look
                is_valid = True

        # 2D BOX: Get pts rect backprojected 
        xmin,ymin,xmax,ymax = det_box2d_list[det_idx]
        box_fov_inds = (pc_image_coord[:,0]<xmax) & \
            (pc_image_coord[:,0]>=xmin) & \
            (pc_image_coord[:,1]<ymax) & \
            (pc_image_coord[:,1]>=ymin)

        if two_frustum and is_valid:
            box_fo_inds_r = (pc_image_coord_r[:,0]<xmax_r) & \
                (pc_image_coord_r[:,0]>=xmin_r) & \
                (pc_image_coord_r[:,1]<ymax_r) & \
                (pc_image_coord_r[:,1]>=ymin_r)
            box_fov_inds = box_fov_inds & img_fov_inds & box_fo_inds_r
        else:
            box_fov_inds = box_fov_inds & img_fov_inds

        pc_in_box_fov = pc_rect[box_fov_inds,:]
        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
        uvdepth = np.zeros((1,3))
        uvdepth[0,0:2] = box2d_center
        uvdepth[0,2] = 20 # some random depth
        box2d_center_rect = calib.project_image_to_rect(uvdepth)
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],
            box2d_center_rect[0,0])
        
        # Pass objects that are too small
        if ymax-ymin<img_height_threshold or \
            len(pc_in_box_fov)<lidar_point_threshold:
            rej +=1
            continue
       
        id_list.append(data_idx)
        type_list.append(det_type_list[det_idx])
        box2d_list.append(det_box2d_list[det_idx])
        prob_list.append(det_prob_list[det_idx])
        input_list.append(pc_in_box_fov)
        frustum_angle_list.append(frustum_angle)

        acc +=1
        print("Number of points :",np.sum(box_fov_inds))
        print( "rej: ",rej,"acc: " ,acc)
    """
    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list,fp)
        pickle.dump(input_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(prob_list, fp)
    """

    if viz:
        import mayavi.mlab as mlab
        for i in range(10):
            p1 = input_list[i]
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,0], p1[:,1], p1[:,2], p1[:,1], mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()

def write_2d_rgb_detection(det_filename, split, result_dir):
    ''' Write 2D detection results for KITTI evaluation.
        Convert from Wei's format to KITTI format. 
        
    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        result_dir: string, folder path for results dumping
    Output:
        None (will write <xxx>.txt files to disk)

    Usage:
        write_2d_rgb_detection("val_det.txt", "training", "results")
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'), split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    # map from idx to list of strings, each string is a line without \n
    results = {} 
    for i in range(len(det_id_list)):
        idx = det_id_list[i]
        typename = det_type_list[i]
        box2d = det_box2d_list[i]
        prob = det_prob_list[i]
        output_str = typename + " -1 -1 -10 "
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        output_str += "-1 -1 -1 -1000 -1000 -1000 -10 %f" % (prob)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close() 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo.')
    parser.add_argument('--gen_train', action='store_true', help='Generate train split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_val', action='store_true', help='Generate val split frustum data with GT 2D boxes')
    parser.add_argument('--gen_val_rgb_detection', action='store_true', help='Generate val split frustum data with RGB detection 2D boxes')
    parser.add_argument('--car_only', action='store_true', help='Only generate cars; otherwise cars, peds and cycs')
    parser.add_argument('--two_frustum', action='store_true', help='Exctracts intersection of two frustums.')
    parser.add_argument('--merge_pseudo', action='store_true', help='Merges Pseudo lidar with Lidar when object has num points less than threshold.')
    parser.add_argument('--pseudo_two_frustum', action='store_true', help='In case of merge_pseudo, if true uses intersection of two frustum.')
    args = parser.parse_args()

    if args.demo:
        demo()
        exit()

    if args.two_frustum:
        print("Two Frustum Mode is Enabled.")

    if args.car_only:
        type_whitelist = ['Car']
        if args.two_frustum: 
            output_prefix = 'two_frustum_caronly_'
        else:
            output_prefix = 'frustum_caronly_'
    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        if args.two_frustum: 
            output_prefix = 'two_frustum_carpedcyc_'
        else:
            output_prefix = 'frustum_carpedcyc_'
    
    if args.gen_train:
        extract_frustum_data(\
            os.path.join(BASE_DIR, 'image_sets/train.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix+'train.pickle'), 
            viz=False, perturb_box2d=True, augmentX=5,
            type_whitelist=type_whitelist, two_frustum=args.two_frustum, 
            merge_pseudo_lidar=args.merge_pseudo, pseudo_two_frustum=args.pseudo_two_frustum)
    
    if args.gen_val:
        extract_frustum_data(\
            os.path.join(BASE_DIR, 'image_sets/val.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix+'val.pickle'),
            viz=False, perturb_box2d=False, augmentX=1,
            type_whitelist=type_whitelist, two_frustum=args.two_frustum, 
            merge_pseudo_lidar=args.merge_pseudo, pseudo_two_frustum=args.pseudo_two_frustum,
            map_outfile=os.path.join(BASE_DIR,output_prefix+'_frustum_mapping.pickle'), gen_mapping=True)
        
    # Still does not support two frustum for extract_frustum_data_rgb_detection
    if args.gen_val_rgb_detection:
        extract_frustum_data_rgb_detection(\
            os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_val.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix+'val_rgb_detection.pickle'),
            viz=False,
            type_whitelist=type_whitelist) 

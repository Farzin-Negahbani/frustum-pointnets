''' Extracts Wrong Box Estimations.
    Author: Onur berk tore & Farzin Negahbani
    Date: May  2020
'''
from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
from datetime import datetime
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider
from train_util import get_batch
import cPickle as pickle


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--model_file', default=None, help='Model file to load e.g. log/model.ckpt')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--frustum_folder', default=None, help='Folder that contains frustums.')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--two_frustum', action='store_true', help='Train on two frustums version[default 1 frustum].')
parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
parser.add_argument('--frustum_map', default=None, help='Frustum Mapping pickle file path')

FLAGS = parser.parse_args()

if not FLAGS.frustum_map is None:
    with open(FLAGS.frustum_map ,'rb') as fp:
        frustum_map = pickle.load(fp)
        
if FLAGS.model_file is None:
    print("Please Specify model file to load weights.")
    print("Exiting...")
    exit()

# Set configurations
EPOCH_CNT = 0
BATCH_SIZE = 1
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
NUM_CHANNEL = 3 if FLAGS.no_intensity else 4 # point feature channel
NUM_CLASSES = 2 # segmentation has two classes

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'extract_mistakes.py'), LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_extract_mistakes.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


# Load Frustum Datasets. 
if FLAGS.two_frustum:
    data_path_val   = os.path.join(ROOT_DIR,'kitti/'+FLAGS.frustum_folder+'/two_frustum_carpedcyc_val.pickle')
else:
    data_path_val   = os.path.join(ROOT_DIR,'kitti/'+FLAGS.frustum_folder+'/frustum_carpedcyc_val.pickle')

print("Loading val frustums from: ",data_path_val)

TEST_DATASET = provider.FrustumDataset(overwritten_data_path= data_path_val,npoints=NUM_POINT, split='val',rotate_to_center=True, one_hot=True)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def main():
    ''' Main function for evaluation and determining mistakes. '''
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = \
                MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)

            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Get model and losses 
            end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,is_training_pl)
            loss = MODEL.get_loss(labels_pl, centers_pl,
                heading_class_label_pl, heading_residual_label_pl,
                size_class_label_pl, size_residual_label_pl, end_points)
            tf.summary.scalar('loss', loss)

            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)

            # Write summaries of bounding box IoU and segmentation accuracies
            iou2ds, iou3ds = tf.py_func(provider.compute_box3d_iou, [\
                end_points['center'], \
                end_points['heading_scores'], end_points['heading_residuals'], \
                end_points['size_scores'], end_points['size_residuals'], \
                centers_pl, \
                heading_class_label_pl, heading_residual_label_pl, \
                size_class_label_pl, size_residual_label_pl], \
                [tf.float32, tf.float32])
            end_points['iou2ds'] = iou2ds 
            end_points['iou3ds'] = iou3ds 
            tf.summary.scalar('iou_2d', tf.reduce_mean(iou2ds))
            tf.summary.scalar('iou_3d', tf.reduce_mean(iou3ds))

            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # load variables from model
        saver.restore(sess, FLAGS.model_file)

        ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'centers_pred': end_points['center'],
               'loss': loss,
               #'train_op': train_op,
               'merged': merged,
               #'step': batch,
               'end_points': end_points}     

        evaluate(sess, ops, test_writer)


def evaluate(sess, ops, test_writer):
    ''' 
    
    '''
    #global frustum map
    global frustum_map

    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET)/BATCH_SIZE

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0

    #for batch_idx in range(num_batches):
    for batch_idx in range(17334):
        
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['labels_pl']: batch_label,
                     ops['centers_pl']: batch_center,
                     ops['heading_class_label_pl']: batch_hclass,
                     ops['heading_residual_label_pl']: batch_hres,
                     ops['size_class_label_pl']: batch_sclass,
                     ops['size_residual_label_pl']: batch_sres,
                     ops['is_training_pl']: is_training}

        summary, loss_val, logits_val, iou2ds, iou3ds = \
            sess.run([ops['merged'],
                ops['loss'], ops['logits'], 
                ops['end_points']['iou2ds'], ops['end_points']['iou3ds']],
                feed_dict=feed_dict)

        test_writer.add_summary(summary)

        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val

        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(batch_label==l)
            total_correct_class[l] += (np.sum((preds_val==l) & (batch_label==l)))
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)
        iou3d_correct_cnt += np.sum(iou3ds>=0.7)

        for i in range(BATCH_SIZE):
            segp = preds_val[i,:]
            segl = batch_label[i,:] 
            part_ious = [0.0 for _ in range(NUM_CLASSES)]
            for l in range(NUM_CLASSES):
                if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0): 
                    part_ious[l] = 1.0 # class not present
                else:
                    part_ious[l] = np.sum((segl==l) & (segp==l)) / \
                        float(np.sum((segl==l) | (segp==l)))

        if iou3ds < 0.7:
            log_string('**** Frustum Number %03d from Image %06d and 2D box %06d****' % (batch_idx,frustum_map[batch_idx]['img_indx'],frustum_map[batch_idx]['obj_indx']))
            log_string('loss: %f' % (loss_val))
            log_string('Frustum segmentation accuracy: %f'% (correct / float(NUM_POINT)))
            log_string('Frustum box IoU (ground/3D): %f / %f' % (iou2ds, iou3ds))
        
    log_string("*********** Summary Results **********")
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval segmentation accuracy: %f'% \
        (total_correct / float(total_seen)))
    log_string('eval segmentation avg class acc: %f' % \
        (np.mean(np.array(total_correct_class) / \
            np.array(total_seen_class,dtype=np.float))))
    log_string('eval box IoU (ground/3D): %f / %f' % \
        (iou2ds_sum / float(num_batches*BATCH_SIZE), iou3ds_sum / \
            float(num_batches*BATCH_SIZE)))
    log_string('eval box estimation accuracy (IoU=0.7): %f' % \
        (float(iou3d_correct_cnt)/float(num_batches*BATCH_SIZE)))

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    log_string(str(datetime.now()))
    main()
    log_string(str(datetime.now()))
    LOG_FOUT.close()

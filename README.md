## Revised Frustum PointNets for 3D Object Detection from RGB-D Data
Under Construction...

## Installation
This code tested under Ubuntu 18.04.
- Python 2.7
- CUDA 10
- CudNN 7.5
- Tensorflow 1.14
- OpenCV 4

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>.There are also some dependencies for a few Python libraries for data processing and visualizations like `cv2`, `mayavi`  etc. It's highly recommended that you have access to GPUs.

To use the Frustum PointNets v2 model, we need access to a few custom Tensorflow operators from PointNet++. The TF operators are included under `models/tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The compile script is written for TF1.4. There is also an option for TF1.2 in the script. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

If we want to evaluate 3D object detection AP (average precision), we need also to compile the evaluation code (by running `compile.sh` under `train/kitti_eval`). Check `train/kitti_eval/README.md` for details.

Some of the demos require `mayavi` library. We have provided a convenient script to install `mayavi` package in Python, a handy package for 3D point cloud visualization. You can check it at `mayavi/mayavi_install.sh`. If the installation succeeds, you should be able to run `mayavi/test_drawline.py` as a simple demo. Note: the library works for local machines and seems do not support remote access with `ssh` or `ssh -X`.

## Usage

Currently, we support training and testing of the Frustum PointNets models as well as evaluating 3D object detection results based on precomputed 2D detector outputs (under `kitti/rgb_detections`). You are welcomed to extend the code base to support your own 2D detectors or feed your own data for network training.

### Prepare Training Data
In this step we convert original KITTI data to organized formats for training our Frustum PointNets. <b>NEW:</b> You can also directly download the prepared data files <a href="https://shapenet.cs.stanford.edu/media/frustum_data.zip" target="_blank">HERE (960MB)</a> -- to support training and evaluation, just unzip the file and move the `*.pickle` files to the `kitti` folder.

Firstly, you need to download the <a href="http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d" target="_blank">KITTI 3D object detection dataset</a>, including left color images, Velodyne point clouds, camera calibration matrices, and training labels. Make sure the KITTI data is organized as required in `dataset/README.md`. You can run `python kitti/kitti_object.py` to see whether data is downloaded and stored properly. If everything is fine, you should see image and 3D point cloud visualizations of the data. 

Then to prepare the data, simply run: (warning: this step will generate around 4.7GB data as pickle files)

    sh scripts/command_prep_data.sh

Basically, during this process, we are extracting frustum point clouds along with ground truth labels from the original KITTI data, based on both ground truth 2D bounding boxes and boxes from a 2D object detector. We will do the extraction for the train (`kitti/image_sets/train.txt`) and validation set (`kitti/image_sets/val.txt`) using ground truth 2D boxes, and also extract data from validation set with predicted 2D boxes (`kitti/rgb_detections/rgb_detection_val.txt`).

You can check `kitti/prepare_data.py` for more details, and run `python kitti/prepare_data.py --demo` to visualize the steps in data preparation.

After the command executes, you should see three newly generated data files under the `kitti` folder. You can run `python train/provider.py` to visualize the training data (frustum point clouds and 3D bounding box labels, in rect camera coordinate).

### Training Frustum PointNets

To start training (on GPU 0) the Frustum PointNets model, just run the following script:

    CUDA_VISIBLE_DEVICES=0 sh scripts/command_train_v1.sh

You can run `scripts/command_train_v2.sh` to trian the v2 model as well. The training statiscs and checkpoints will be stored at `train/log_v1` (or `train/log_v2` if it is a v2 model). Run `python train/train.py -h` to see more options of training. 

<b>NEW:</b> We have also prepared some pretrained snapshots for both the v1 and v2 models. You can find them <a href="https://shapenet.cs.stanford.edu/media/frustum_pointnets_snapshots.zip" target="_blank">HERE (40MB)</a> -- to support evaluation script, you just need to unzip the file and move the `log_*` folders to the `train` folder.

### Evaluation
To evaluate a trained model (assuming you already finished the previous training step) on the validation set, just run:

    CUDA_VISIBLE_DEVICES=0 sh scripts/command_test_v1.sh

Similarly, you can run `scripts/command_test_v2.sh` to evaluate a trained v2 model. The script will automatically evaluate the Frustum PointNets on the validation set based on precomputed 2D bounding boxes from a 2D detector (not released here), and then run the KITTI offline evaluation scripts to compute precision recall and calcuate average precisions for 2D detection, bird's eye view detection and 3D detection.

Currently there is no script for evaluation on test set, yet it is possible to do it by yourself. To evaluate on the test set, you need to get outputs from a 2D detector on KITTI test set, store it as something in `kitti/rgb_detections`. Then, you need to prepare test set frustum point clouds for the test set, by modifying the code in `kitti/prepare_data.py`. Then you can modify test scripts in `scripts` by changing the data path, idx path and output file name. For our test set results reported, we used the entire `trainval` set for training.

## License
Our code is released under the Apache 2.0 license (see LICENSE file for details).

## References
* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017 Oral Presentation). Code and data: <a href="https://github.com/charlesq34/pointnet">here</a>.
* <a href="http://stanford.edu/~rqi/pointnet2" target="_black">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017). Code and data: <a href="https://github.com/charlesq34/pointnet2">here</a>.

* <a href="http://stanford.edu/~rqi/frustum-pointnets/" target="_black">Frustum PointNets for 3D Object Detection from RGB-D Data</a> by Qi et al. (CVPR) 2018 . Code and data: <a href="https://github.com/charlesq34/frustum-pointnets">here</a>.

### Todo
- [x] Adopting code with CUDA 10
- [ ] Add revised frustum extraction code
- [ ] Add a demo script to run inference of Frustum PointNets based on raw input data.
- [ ] Add related scripts for SUNRGBD dataset

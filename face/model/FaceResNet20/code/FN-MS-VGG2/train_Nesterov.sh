#!/bin/bash
# Usage:
# ./code/train.sh GPU
#
# Example:
# ./code/train.sh 0,1,2,3
GPU_ID=$1
$CAFFE_HOME/build/tools/caffe train -solver code/FN-MS-VGG2/FaceResNet20_step1_Nesterov_solver.prototxt -gpu ${GPU_ID}
$CAFFE_HOME/build/tools/caffe train -solver code/FN-MS-VGG2/FaceResNet20_step2_Nesterov_solver.prototxt -weights result/FN-MS-VGG2/FaceResNet20_Nesterov_step1_iter_60000.caffemodel -gpu ${GPU_ID}
$CAFFE_HOME/build/tools/caffe train -solver code/FN-MS-VGG2/FaceResNet20_step3_Nesterov_solver.prototxt -weights result/FN-MS-VGG2/FaceResNet20_Nesterov_step2_iter_100000.caffemodel -gpu ${GPU_ID}
$CAFFE_HOME/build/tools/caffe train -solver code/FN-MS-VGG2/FaceResNet20_step4_Nesterov_solver.prototxt -weights result/FN-MS-VGG2/FaceResNet20_Nesterov_step3_iter_120000.caffemodel -gpu ${GPU_ID}
$CAFFE_HOME/build/tools/caffe train -solver code/FN-MS-VGG2/FaceResNet20_step5_Nesterov_solver.prototxt -weights result/FN-MS-VGG2/FaceResNet20_Nesterov_step4_iter_120000.caffemodel -gpu ${GPU_ID}

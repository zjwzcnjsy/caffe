#!/bin/bash
# Usage:
# ./code/train.sh GPU
#
# Example:
# ./code/train.sh 0,1,2,3
GPU_ID=$1
$CAFFE_HOME/build/tools/caffe train -solver code/FN-MS-VGG2/FaceResNet20_step1_adam_solver.prototxt -gpu ${GPU_ID}
$CAFFE_HOME/build/tools/caffe train -solver code/FN-MS-VGG2/FaceResNet20_step2_adam_solver.prototxt -weights result/FN-MS-VGG2/FaceResNet20_step1_adam_iter_60000.caffemodel -gpu ${GPU_ID}
$CAFFE_HOME/build/tools/caffe train -solver code/FN-MS-VGG2/FaceResNet20_step3_adam_solver.prototxt -weights result/FN-MS-VGG2/FaceResNet20_step2_adam_iter_50000.caffemodel -gpu ${GPU_ID}
$CAFFE_HOME/build/tools/caffe train -solver code/FN-MS-VGG2/FaceResNet20_step4_adam_solver.prototxt -weights result/FN-MS-VGG2/FaceResNet20_step3_adam_iter_80000.caffemodel -gpu ${GPU_ID}
$CAFFE_HOME/build/tools/caffe train -solver code/FN-MS-VGG2/FaceResNet20_step5_adam_solver.prototxt -weights result/FN-MS-VGG2/FaceResNet20_step4_adam_iter_80000.caffemodel -gpu ${GPU_ID}

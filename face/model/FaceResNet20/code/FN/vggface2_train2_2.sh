#!/bin/bash
# Usage:
# ./code/train.sh GPU
#
# Example:
# ./code/train.sh 0,1,2,3
GPU_ID=$1
$CAFFE_HOME/build/tools/caffe train -solver code/FN/FaceResNet20_vggface2_step3_2_solver.prototxt -weights result/FN/FaceResNet20_vggface2_step3_2_r2_iter_60000.caffemodel -gpu ${GPU_ID}

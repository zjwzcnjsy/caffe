#!/bin/bash
# Usage:
# ./code/train.sh GPU
#
# Example:
# ./code/train.sh 0,1,2,3
GPU_ID=$1
$CAFFE_HOME/build/tools/caffe train -solver code/FN2/FaceResNet20_vggface2_step4_solver.prototxt -weights result/FN2/FaceResNet20_vggface2_step3_iter_60000.caffemodel -gpu ${GPU_ID}
$CAFFE_HOME/build/tools/caffe train -solver code/FN2/FaceResNet20_vggface2_step5_solver.prototxt -weights result/FN2/FaceResNet20_vggface2_step4_iter_60000.caffemodel -gpu ${GPU_ID}

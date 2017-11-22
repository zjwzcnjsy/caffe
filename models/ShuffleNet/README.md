# ShuffleNet
ShuffleNet的caffe实现，详细信息请了解论文：
["ShuffleNet: An Extremely Efficient Convolutional
Neural Network for Mobile Devices" by Xiangyu Zhang et. al. 2017](https://arxiv.org/pdf/1707.01083.pdf)。

代码来自farmingyard的实现(https://github.com/farmingyard/ShuffleNet)。

## How to use?
#### caffe.proto:
```
message LayerParameter {
...
optional ShuffleChannelParameter shuffle_channel_param = 164;
...
}
...
message ShuffleChannelParameter {
  optional uint32 group = 1[default = 1]; // The number of group
}
```

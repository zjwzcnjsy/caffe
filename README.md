## 安装
``` bash
git clone http://192.168.1.105:8888/jsy/caffe.git
cd caffe
cp Makefile.config.example Makefile.config
```

修改Makfile.config文件里的相关配置，比如python等。

然后编译：
``` bash
make all -j12
```

编译完成的结果在`build`目录下，可执行程序`caffe`在`build/tools`下。
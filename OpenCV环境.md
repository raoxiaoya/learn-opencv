##### 演化流程

`OpenCV  --> 传统的机器学习SIFT --> 神经网络CNN`

计算机视觉也由诸如梯度方向直方图（Histogram of Gradient， HOG）以及尺度不变特征变换（Scale-Invariant Feature Transform， SIFT）等传统的手办特征（Hand-Crafted Feature）与浅层模型的组合逐渐转向了以卷积神经网络（Convolutional Neural Network， CNN）为代表的深度学习模型。

OpenCV的全称是Open Source Computer Vision Library，是一个跨平台的计算机视觉库。OpenCV是由英特尔公司发起并参与开发，以BSD许可证授权发行，可以在商业和研究领域中免费使用。OpenCV可用于开发实时的图像处理、计算机视觉以及模式识别程序。该程序库也可以使用英特尔公司的IPP进行加速处理。



##### 关于OpenCV

https://github.com/opencv/opencv/tree/4.7.0

https://opencv.org/releases/

https://docs.opencv.org/4.7.0/  当前稳定版本 4.7.0

opencv论坛 https://forum.opencv.org/



#### python配置OpenCV

```bash
conda create -n opencv4.10 python==3.8
conda activate opencv4.10
pip install opencv-python==4.10.0.84 opencv-contrib-python==4.10.0.84 numpy==1.20.3 pytesseract==0.3.13 pillow=9.0.0
```




#### golang配置OpenCV

官网教程 https://gocv.io/getting-started/windows/

要使用gocv，需要在本机上编译安装opencv，你单纯的下载安装包还不行，比如安装包`https://github.com/opencv/opencv/releases/download/4.7.0/opencv-4.7.0-windows.exe`，因为你把安装包解压了之后就能看到它只有python和java的bingdings。

我问了一下DeepSeek：为什么gocv要编译安装opencv，而cv2却不需要编译安装opencv呢？

其回答如下：

`gocv` 和 `cv2`（通常指的是Python的OpenCV接口）在处理与OpenCV库的关系时采用了不同的方法，这导致了它们在使用前是否需要编译安装OpenCV有所不同。

1. **gocv**：`gocv` 是一个用于Go语言的计算机视觉库，它提供了一个Go语言接口来调用OpenCV的功能。由于Go语言和C++（OpenCV是用C++写的）之间存在差异，`gocv` 通过CGO等机制实现Go代码对C++库的调用。这意味着当你在使用`gocv`之前，可能需要先确保系统中已经正确编译安装了对应版本的OpenCV库。因为`gocv`依赖于底层的OpenCV C++ API，并且需要针对特定平台编译相应的绑定。
2. **cv2 (Python-OpenCV)**：另一方面，当使用Python中的OpenCV（通常通过`import cv2`来使用），大多数情况下你只需要通过pip等包管理工具直接安装预编译的二进制文件（如`pip install opencv-python`）。这些预编译的包已经包含了所有必要的OpenCV组件，包括其C++部分，并且被配置为可以直接在Python环境中使用。因此，用户不需要自己手动编译OpenCV源码。

总结来说，区别主要在于编程语言及其生态系统的特点：`gocv`作为Go语言的接口，需要开发者手动设置好C++环境以及OpenCV的编译安装；而`cv2`利用了Python丰富的包管理和预编译分发机制，简化了用户的使用流程，使得用户无需关心底层细节即可快速开始使用OpenCV的功能。



gocv 与 golang 的版本以及opencv的版本要匹配上，比如 go1.18 安装的 gocv-v0.33.0，它对应的是 OpenCV-4.7.0，成功安装了，也可以运行，后来 go 升级到 1.23，于是就无法使用。

对应的仓库 `https://github.com/hybridgroup/gocv` ，版本为`v0.33.0`，大量示例在`gocv/cmd`目录下。

先安装`MinGW-W64 v8.1.0`和`CMake`并添加到环境变量。

```bash
> cmake --version
cmake version 3.27.0-rc3

CMake suite maintained and supported by Kitware (kitware.com/cmake).

> gcc -v
gcc version 8.1.0 (x86_64-posix-seh-rev0, Built by MinGW-W64 project)
```

下载包 `go get gocv.io/x/gocv@v0.33.0`

提前打开梯子。

查看安装程序 `%GOPATH%\pkg\mod\gocv.io\x\gocv@v0.40.0\win_build_opencv.cmd`，其过程倒是很简单，在安装的过程中发现通过此脚本下载安装包要比在浏览器上下载慢很多，于是可以先手动下载。

```bash
https://github.com/opencv/opencv/archive/4.7.0.zip
https://github.com/opencv/opencv_contrib/archive/4.7.0.zip
```

然后修改`win_build_opencv.cmd`文件，去掉下载部分的代码即可。其次此文件里面写死了安装到在`C:\opencv`，版本为`OpenCV-4.7.0`。

使用`管理员身份`执行`cmd`，进入`gocv`，运行`win_build_opencv.cmd`。需要一个小时左右。

然后将`C:\opencv\build\install\x64\mingw\bin`添加到环境变量。

验证安装是否成功，进入到`GOPATH`目录下找到`gocv`，执行`go run cmd\version\main.go`。输出如下说明安装成功

```bash
gocv version: 0.33.0
opencv lib version: 4.7.0
```

注意，opencv的功能是大量依赖于你本地环境的，在安装opencv的时候会看到大量的`not found`，然后它就将对应的功能设置为`off`，于是你在使用的时候就会报错，如果的确需要某功能，需要先装上对应的依赖，然后重新编译即可。

```bash
rgbd: Eigen support is disabled. Eigen is Required for Posegraph optimization
```

在安装的时候，有的系统会报错`D:\Program`不是一个可运行程序，那是因为MinGW被安装在了`D:\Program Files`目录下，在命令行模式下，这个空格被解析为前面是程序，后面是参数，因此报错，我修改了`win_build_opencv.cmd`文件，将`mingw32-make`写成了绝对路径，并加上了双引号，如下：`D:\"Program Files"\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\mingw32-make`，但是架不住还有隐藏的调用也报这个错，然后我的CMake是安装在`C:\Program Files`下面，这居然不报错，于是我就简单的将`D:\Program Files\mingw-w64` 移到了 `C:\Program Files\mingw-w64`，然后环境变量也跟着改了一下，居然就可以了。

学习文档：
https://blog.csdn.net/qq_15698613/category_9292368.html

https://github.com/hybridgroup/gocv/cmd

https://blog.csdn.net/youcans/article/details/125112487

Go数学库：gonum
对应的仓库 github.com/gonum/gonum
安装 go get -u gonum.org/v1/gonum/...
import "gonum.org/v1/gonum"
文档 godoc -http=:6060

第三方绘图库 plot
go get -u gonum.org/v1/plot/...
gonum/plot 示例  https://github.com/gonum/plot/wiki/Example-plots

第三方绘图库 go-echarts 好用
go get -u github.com/go-echarts/go-echarts/...
go get -u github.com/go-echarts/go-echarts/v2/...
import "github.com/go-echarts/go-echarts/v2/charts"


#### OpenCV概述

各个模块的功能：

`calib3d`
其实就是就是Calibration（校准）加3D这两个词的组合缩写。这个模块主要是相机校准和三维重建相关的内容。基本的多视角几何算法，单个立体摄像头标定，物体姿态估计，立体相似性算法，3D信息的重建等等。

`core`
核心功能模块，包含如下内容：

- OpenCV基本数据结构
- 动态数据结构
- 绘图函数
- 数组操作相关函数
- 辅助功能与系统函数和宏
- 与OpenGL的互操作

`imgproc`

Image和Processing这两个单词的缩写组合。图像处理模块，这个模块包含了如下内容：

- 线性和非线性的图像滤波

- 图像的几何变换
- 其它（Miscellaneous）图像转换
- 直方图相关
- 结构分析和形状描述
- 运动分析和对象跟踪
- 特征检测
- 目标检测等内容

`features2d`

也就是Features2D， 2D功能框架 ，包含如下内容：

- 特征检测和描述

- 特征检测器（Feature Detectors）通用接口
- 描述符提取器（Descriptor Extractors）通用接口
- 描述符匹配器（Descriptor Matchers）通用接口
- 通用描述符（Generic Descriptor）匹配器通用接口
- 关键点绘制函数和匹配功能绘制函数

`flann`

 Fast Library for Approximate Nearest Neighbors，高维的近似近邻快速搜索算法库，包含两个部分

- 快速近似最近邻搜索
- 聚类

`highgui`

也就是high gui，高层GUI图形用户界面，包含媒体的I / O输入输出，视频捕捉、图像和视频的编码解码、图形交互界面的接口等内容

`ml`


Machine Learning，机器学习模块， 基本上是统计模型和分类算法，包含如下内容：

- 统计模型 （Statistical Models）
- 一般贝叶斯分类器 （Normal Bayes Classifier）
- K-近邻 （K-NearestNeighbors）
- 支持向量机 （Support Vector Machines）
- 决策树 （Decision Trees）
- 提升（Boosting）
- 梯度提高树（Gradient Boosted Trees）
- 随机树 （Random Trees）
- 超随机树 （Extremely randomized trees）
- 期望最大化 （Expectation Maximization）
- 神经网络 （Neural Networks）
- MLData

`objdetect`

目标检测模块，包含Cascade Classification（级联分类）和Latent SVM这两个部分。

`photo`

也就是Computational Photography，包含图像修复和图像去噪两部分

`stitching`

images stitching，图像拼接模块，包含如下部分：

- 拼接流水线
- 特点寻找和匹配图像
- 估计旋转
- 自动校准
- 图片歪斜
- 接缝估测
- 曝光补偿
- 图片混合

`video`


视频分析组件，该模块包括运动估计，背景分离，对象跟踪等视频处理相关内容。






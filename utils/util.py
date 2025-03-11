import cv2
import matplotlib.pyplot as plt

def showImage(title, image, shouldWaitKey=False):
    cv2.imshow(title, image)
    if shouldWaitKey:
        cv2.waitKey(0)

def showMultipleImage(size: tuple, images: list):
    '''
    plt 展示多张图片

    1、OpenCV 和 matplotlib 中的彩色图像都是 Numpy 多维数组。但 OpenCV 使用 BGR 格式，颜色分量按照蓝/绿/红的次序排列，而 matplotlib 使用 RGB 格式，颜色分量按照红/绿/蓝的次序排序。因此用 plt.imshow() 显示 OpenCV 彩色图像时，先要进行颜色空间转换，将Numpy 多维数组按照红/绿/蓝的次序排序，否则色彩不对。

    2、plt.imshow() 可以直接显示 OpenCV 灰度图像，不需要格式转换，但需要使用 cmap=‘gray’ 进行参数设置，否则色彩不对。
    '''
    plt.rcParams['font.sans-serif'] = ['FangSong']  # 支持中文标签，仿宋字体
    for k, v in enumerate(images):
        '''
        nrows: 子图的行数。
        ncols: 子图的列数。
        index: 当前子图的位置索引，从1开始计数，按照从左到右、从上到下的顺序编号。
        '''
        plt.subplot(size[0], size[1], k+1), plt.title(v['title']), plt.axis('off')
        if v['cmap'] != '':
            plt.imshow(v['image'], cmap=v['cmap'])
        else:
            plt.imshow(v['image'])

    plt.show()
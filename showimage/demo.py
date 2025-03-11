import cv2
from utils import util
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageSequence, ImageDraw, ImageFont
import requests
import os
import math

# 参考教程：https://blog.csdn.net/youcans/article/details/125112487


def run1():
    img = cv2.imread("showimage/cat.jpg")
    cv2.imshow("show image", img)
    cv2.moveWindow("show image", 800, 300)
    cv2.waitKey(0)


def run3():
    '''
    图像腐蚀与膨胀
    '''
    img = cv2.imread("showimage/cat.jpg", cv2.IMREAD_COLOR)
    util.showImage("原图", img)

    dst = cv2.erode(img, (15, 15))  # 腐蚀操作，黑色变大
    util.showImage("图像腐蚀-后", dst)

    dst2 = cv2.dilate(img, (15, 15))  # 膨胀操作，白色变大
    util.showImage("图像膨胀-后", dst2, True)


def run4():
    img = cv2.imread("showimage/cat.jpg", cv2.IMREAD_COLOR)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将原始图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 单通道

    util.showMultipleImage((2, 2), [
        {"title": "1. RGB 格式(mpl)", "image": imgRGB, "cmap": ''},
        {"title": "2. BGR 格式(OpenCV)", "image": img, "cmap": ''},
        {"title": "3. 设置 Gray 参数", "image": gray, "cmap": 'gray'},
        {"title": "4. 未设置 Gray 参数", "image": gray, "cmap": ''},
    ])


def run5():
    '''
    边缘检测
    '''
    img = cv2.imread("showimage/cat.jpg", cv2.IMREAD_COLOR)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将原始图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 单通道

    # 先用3*3内核来降噪，模糊处理
    edge = cv2.blur(gray, (3, 3), gray, anchor=(-1, -1),
                    borderType=cv2.BORDER_DEFAULT)

    # 运行Canny算子
    canny = cv2.Canny(edge, 3, 9)

    util.showMultipleImage((2, 2), [
        {"title": "imgRGB", "image": imgRGB, "cmap": ''},
        {"title": "gray", "image": gray, "cmap": 'gray'},
        {"title": "edge", "image": edge, "cmap": 'gray'},
        {"title": "canny", "image": canny, "cmap": 'gray'},
    ])


def run6():
    '''
    卷积运算
    '''
    # src = np.array([
    #     [181, 8, 127, 14, 208, 158],
    #     [144, 59, 51, 179, 228, 118],
    #     [160, 7, 212, 101, 242, 240],
    #     [151, 136, 22, 95, 63, 47],
    #     [250, 215, 161, 189, 66, 234],
    #     [146, 166, 70, 201, 136, 201]
    # ])

    src = cv2.imread("showimage/cat.jpg", cv2.IMREAD_COLOR)

    kernel = np.array([
        [0.1111, 0, 0.1111],
        [0.1111, 0, 0.1111],
        [0.1111, 0, 0.1111]
    ])
    '''
    src: 输入图像。
    ddepth: 输出图像的深度。如果设置为 -1，则输出图像的深度与输入图像相同。
    kernel: 卷积核（或称为滤波器），通常是一个 numpy.ndarray 类型的小矩阵。
    dst: 可选参数，输出图像，其大小和通道数与输入图像相同。
    anchor: 锚点，默认值为 (-1, -1)，表示锚位于内核的中心。
    delta: 可选参数，从过滤结果中添加到目标图像上的值，默认为 0。
    borderType: 边界填充类型，默认使用的是 BORDER_DEFAULT。
    '''
    dst = cv2.filter2D(src, -1, kernel)

    print(dst.shape)


def run7():
    '''
    扩充边界
    '''
    src = cv2.imread("showimage/cat.jpg", cv2.IMREAD_COLOR)
    dst = cv2.copyMakeBorder(
        src, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 255, 0))
    util.showImage("dst", dst, True)


def run8():
    '''
    图像翻转
    '''
    src = cv2.imread("showimage/cat.jpg", cv2.IMREAD_COLOR)
    '''
    0 - 沿着水平线翻转
	1 - 沿着垂直线翻转
	-1 - 沿着水平和垂直线翻转
    '''
    dst = cv2.flip(src, 1)
    util.showImage("src", src)
    util.showImage("dst", dst, True)


def run9():
    '''
    图像阈值化
    '''
    src = cv2.imread("showimage/cat.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    '''
    src: 输入图像，应为灰度图像（单通道）。
    thresh: 阈值，用于分类像素值。
    maxval: 当达到阈值时，赋予像素的最大值（主要与某些阈值类型有关）。
    type: 阈值处理的类型。OpenCV 提供了多种不同的阈值方法。
    '''
    ret, dst = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    util.showImage("src", src)
    util.showImage("dst", dst, True)


def run10():
    '''
    读物GIF动图
    '''
    # 使用Pillow打开GIF
    gif = Image.open("showimage/image15.gif")

    # 遍历GIF的每一帧
    for frame in ImageSequence.Iterator(gif):
        # 将当前帧转换为RGB模式（如果需要）
        frame = frame.convert("RGB")

        # 将Pillow图像转换为NumPy数组/OpenCV格式
        opencv_image = np.array(frame)

        # 转换颜色通道顺序（因为Pillow使用RGB而OpenCV使用BGR）
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

        # 显示图像
        cv2.imshow('GIF', opencv_image)

        # 控制帧率，这里假设GIF的帧率为25fps，可根据实际情况调整
        # 等待40毫秒或按下'q'键退出
        # waitkey参数为毫秒
        if cv2.waitKey(3000) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run11():
    '''
    读取视频文件
    '''
    cap = cv2.VideoCapture("showimage/video1.mp4")
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imshow("video", frame)
        if cv2.waitKey(3000) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)


def run12():
    '''
    读取网络图片
    '''
    response = requests.get(
        "https://videoactivity.bookan.com.cn/ac_1_1687763619_793.jpg")

    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

    dst = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)


def run13():
    '''
    图片裁剪，使用鼠标选择感兴趣的区域
    '''
    img = cv2.imread("showimage/cat.jpg", cv2.IMREAD_COLOR)
    rect = cv2.selectROI(img)
    xmin, ymin, w, h = rect
    dst = img[ymin:ymin+h, xmin:xmin+w].copy()

    cv2.imshow("dst", dst)
    cv2.waitKey(0)


def run14():
    '''
    拆分通道，合并通道
    '''
    img = cv2.imread("showimage/cat.jpg", cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)
    print(b[0][:12])
    # [53 54 54 55 55 56 56 56 58 59 62 65]
    print(g[0][:12])
    # [55 56 56 57 57 58 58 58 59 60 63 66]
    print(r[0][:12])
    # [ 95  96  96  97  98  99  99  99 103 104 107 110]

    imgMerge = cv2.merge((b, g, r))


def run15():
    height, width, channels = 400, 300, 3  # 行/高度, 列/宽度, 通道数
    imgEmpty = np.empty((height, width, channels), np.uint8)  # 创建空白数组
    imgBlack = np.zeros((height, width, channels), np.uint8)  # 创建黑色图像 RGB=0
    imgWhite = np.ones((height, width, channels),
                       np.uint8) * 255  # 创建白色图像 RGB=255
    # (2) 创建相同形状的多维数组
    img1 = cv2.imread("showimage/cat.jpg", flags=1)  # flags=1 读取彩色图像(BGR)
    imgBlackLike = np.zeros_like(img1)  # 创建与 img1 相同形状的黑色图像
    imgWhiteLike = np.ones_like(img1) * 255  # 创建与 img1 相同形状的白色图像
    # (3) 创建彩色随机图像 RGB=random
    randomByteArray = bytearray(os.urandom(height * width * channels))
    flatNumpyArray = np.array(randomByteArray)
    imgRGBRand = flatNumpyArray.reshape(height, width, channels)
    # (4) 创建灰度图像
    imgGrayWhite = np.ones((height, width), np.uint8) * 255  # 创建白色图像 Gray=255
    imgGrayBlack = np.zeros((height, width), np.uint8)  # 创建黑色图像 Gray=0
    imgGrayEye = np.eye(width)  # 创建对角线元素为1 的单位矩阵
    randomByteArray = bytearray(os.urandom(height*width))
    flatNumpyArray = np.array(randomByteArray)
    imgGrayRand = flatNumpyArray.reshape(height, width)  # 创建灰度随机图像 Gray=random

    print("Shape of image: gray {}, RGB {}".format(
        imgGrayRand.shape, imgRGBRand.shape))
    cv2.imshow("DemoGray", imgGrayRand)  # 在窗口显示 灰度随机图像
    cv2.imshow("DemoRGB", imgRGBRand)  # 在窗口显示 彩色随机图像
    cv2.imshow("DemoBlack", imgBlack)  # 在窗口显示 黑色图像
    key = cv2.waitKey(0)  # 等待按键命令


def run16():
    '''
    像素的读取和编辑
    '''
    # 1.13 Numpy 获取和修改像素值
    img1 = cv2.imread("showimage/cat.jpg", flags=1)  # flags=1 读取彩色图像(BGR)
    x, y = 10, 10  # 指定像素位置 x, y

    # (1) 直接访问数组元素，获取像素值(BGR)
    pxBGR = img1[x, y]  # 访问数组元素[x,y], 获取像素 [x,y] 的值
    print("x={}, y={}\nimg[x,y] = {}".format(x, y, img1[x, y]))
    # (2) 直接访问数组元素，获取像素通道的值
    print("img[{},{},ch]:".format(x, y))
    for i in range(3):
        print(img1[x, y, i], end=' ')  # i=0,1,2 对应 B,G,R 通道
    # (3) img.item() 访问数组元素，获取像素通道的值
    print("\nimg.item({},{},ch):".format(x, y))
    for i in range(3):
        print(img1.item(x, y, i), end=' ')  # i=0,1,2 对应 B,G,R 通道

    # (4) 修改像素值：img.itemset() 访问数组元素，修改像素通道的值
    ch, newValue = 0, 255
    print("\noriginal img[x,y] = {}".format(img1[x, y]))
    img1.itemset((x, y, ch), newValue)  # 将 [x,y,channel] 的值修改为 newValue
    print("updated img[x,y] = {}".format(img1[x, y]))


def run17():
    '''
    垂直拼接，水平拼接
    '''
    # 创建红，绿，蓝三张图片
    height, width = 200, 200
    black = np.zeros((height, width, 3), np.uint8)

    blue = black.copy()
    green = black.copy()
    red = black.copy()

    # BGR
    blue[:, :, 0] = 255
    green[:, :, 1] = 255
    red[:, :, 2] = 255

    dst1 = cv2.hconcat((blue, green, red))
    dst2 = cv2.vconcat((blue, green, red))

    cv2.imshow("dst1", dst1)
    cv2.imshow("dst2", dst2)
    cv2.waitKey(0)


def run18():
    '''
    图像加法，不同尺寸相加，即将小图像叠加到大图的特定位置上
    '''
    iconImage = cv2.imread("showimage/gongfu.jpg", cv2.IMREAD_COLOR)
    iconSize = iconImage.shape
    backImage = cv2.imread("showimage/girl.jpg", cv2.IMREAD_COLOR)
    alpha, beta, gamma = 0.5, 0.5, 0.0
    backRegin = backImage[10:10+iconSize[0], 10:10+iconSize[1]]

    '''
    add, addWeight 要求尺寸和通道数完全一致

    alpha, beta 分别表示第一个和第二个的权重，0-1之间，通常 alpha+beta=1

    gamma 灰度系数，图像校正的偏移量，用于调节亮度，dst = src1 * alpha + src2 * beta + gamma
    '''
    dst = cv2.addWeighted(iconImage, alpha, backRegin, beta, gamma)

    backImage[10:10+iconSize[0], 10:10+iconSize[1]] = dst

    cv2.imshow("backImage", backImage)
    cv2.waitKey(0)


def run19():
    '''
    圆形蒙版
    '''
    img1 = cv2.imread("showimage/girl.jpg", cv2.IMREAD_COLOR)
    # 蒙版尺寸与源图像的尺寸要一致
    # 蒙版为 8位灰度格式，0表示过滤掉，255表示接受
    Mask1 = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)
    # 绘制圆形蒙版
    cv2.circle(Mask1, (250, 150), 50, (255, 255, 255), -1)

    # 使img1与一个黑色图形相加，实际上没有发生任何变化。
    # 再利用 cv2.add 方法提供的蒙版来抠图。
    imgAddMask1 = cv2.add(img1, np.zeros(
        np.shape(img1), dtype=np.uint8), mask=Mask1)

    cv2.imshow("imgAddMask1", imgAddMask1)
    cv2.waitKey(0)


def run20():
    '''
    把 jpg 中的图标扣出来粘贴到另一张图片上
    '''
    # logo
    logoSrc = cv2.imread("showimage/logo.jpg", cv2.IMREAD_COLOR)
    # 灰度化
    logoGray = cv2.cvtColor(logoSrc, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, logoBin = cv2.threshold(logoGray, 175, 255, cv2.THRESH_BINARY)
    # 黑白颠倒
    logoInv = cv2.bitwise_not(logoBin)
    # 抠图
    logo = cv2.add(logoSrc, np.zeros(
        np.shape(logoSrc), dtype=np.uint8), mask=logoInv)

    # 目标
    PageSrc = cv2.imread("showimage/girl.jpg", cv2.IMREAD_COLOR)
    regin = PageSrc[:logo.shape[0], :logo.shape[1]]

    # 从原图上扣掉 logo 的像素点
    regin2 = cv2.add(regin, np.zeros(
        np.shape(logoSrc), dtype=np.uint8), mask=logoBin)

    regin3 = cv2.add(logo, regin2)
    PageSrc[:logo.shape[0], :logo.shape[1]] = regin3
    cv2.imshow("PageSrc", PageSrc)
    cv2.waitKey(0)


def run21():
    '''
    图像中添加文本
    '''
    PageSrc = cv2.imread("showimage/girl.jpg", cv2.IMREAD_COLOR)

    '''
    fontFace 字体
	fontScale 字体缩放比例
	tnickness 线条宽度
	bottomLeftOrigin true表示数据原点位于左下角，False表示位于左上角，true是上下颠倒的
	不支持中文（包括中文标点符号）
    '''
    dst = cv2.putText(PageSrc, "OpenCV 2023, showimage/girl.jpg", (100, 100),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA, False)

    '''
    在图像中添加中文字符，可以使用 python+opencv+PIL 实现，
    或使用 python+opencv+freetype 实现。
    '''
    imgPIL = Image.fromarray(cv2.cvtColor(PageSrc, cv2.COLOR_BGR2RGB))
    text = "OpenCV2021, 中文字体"
    pos = (100, 100)  # (left, top)，字符串左上角坐标
    color = (0, 255, 0)  # 字体颜色
    textSize = 40
    drawPIL = ImageDraw.Draw(imgPIL)

    fontText = ImageFont.truetype(
        "c:/windows/fonts/msyh.ttc", textSize, encoding="utf-8")
    drawPIL.text(pos, text, color, font=fontText)
    imgPutText = cv2.cvtColor(np.asarray(imgPIL), cv2.COLOR_RGB2BGR)

    cv2.imshow("imgPutText", imgPutText)
    cv2.waitKey(0)


def run22():
    '''
    图像平移

    二维空间的变换矩阵为 2行3列

    平移
    vec1 = M·vec0 ； M 为变换矩阵，跟 OpenGL 一样，dx为正则向右，dy为正则向下

        | 1 0 dx |			| x	|			 | x + dx|
    M = | 0 1 dy |     vec =| y |      vec1 =| y + dy|
        | 0 0 1  |			| 1 |			 | 1     |
    '''
    src = cv2.imread("showimage/gongfu.jpg", cv2.IMREAD_COLOR)

    '''
    src: 变换操作的输入图像
    M: 仿射变换矩阵,2行3列
    dsize:  输出图像的大小,二元元组 (width, height)
    dst: 变换操作的输出图像,可选项
    flags: 插值方法,整型int,可选项
    cv2.INTER_LINEAR: 线性插值,默认选项
    cv2.INTER_NEAREST: 最近邻插值
    cv2.INTER_AREA: 区域插值
    cv2.INTER_CUBIC: 三次样条插值
    cv2.INTER_LANCZOS4: Lanczos 插值
    borderMode: 边界像素方法,整型int,可选项,默认值为 cv2.BORDER_REFLECT
    borderValue: 边界填充值,可选项,默认值为 0(黑色填充)
    返回值: dst,变换操作的输出图像,ndarray 多维数组
    '''
    dst = cv2.warpAffine(src, np.float32(
        [[1, 0, 10], [0, 1, 50]]), (src.shape[1], src.shape[0]), borderValue=(255, 255, 255))

    util.showMultipleImage((1, 2), [
        {"title": "src", "image": cv2.cvtColor(
            src, cv2.COLOR_BGR2RGB), "cmap": ''},
        {"title": "Translational", "image": cv2.cvtColor(
            dst, cv2.COLOR_BGR2RGB), "cmap": ''},
    ])


def run23():
    '''
    图像旋转
    '''
    src = cv2.imread("showimage/gongfu.jpg", cv2.IMREAD_COLOR)
    img90 = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
    img180 = cv2.rotate(src, cv2.ROTATE_180)
    img270 = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)
    util.showMultipleImage((1, 4), [
        {"title": "src", "image": cv2.cvtColor(
            src, cv2.COLOR_BGR2RGB), "cmap": ''},
        {"title": "img90", "image": cv2.cvtColor(
            img90, cv2.COLOR_BGR2RGB), "cmap": ''},
        {"title": "img180", "image": cv2.cvtColor(
            img180, cv2.COLOR_BGR2RGB), "cmap": ''},
        {"title": "img270", "image": cv2.cvtColor(
            img270, cv2.COLOR_BGR2RGB), "cmap": ''},
    ])


def run24():
    '''
    图像旋转特定角度

    围绕着左上角（0,0）点旋转

        | cosθ  -sinθ  0 |
    M = | sinθ  cosθ   0 |
        | 0     0      1 |
    '''
    src = cv2.imread("showimage/gongfu.jpg", cv2.IMREAD_COLOR)
    θ = math.pi / 6
    m = np.float32([
        [math.cos(θ), -math.sin(θ), 0],
        [math.sin(θ), math.cos(θ), 0],
    ])
    dst = cv2.warpAffine(src, m, (src.shape[1], src.shape[0]),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    util.showMultipleImage((1, 2), [
        {"title": "src", "image": cv2.cvtColor(
            src, cv2.COLOR_BGR2RGB), "cmap": ''},
        {"title": "Translational", "image": cv2.cvtColor(
            dst, cv2.COLOR_BGR2RGB), "cmap": ''},
    ])


def run25():
    '''
    图像旋转

    围绕着任意点旋转，先将该点移动到（0,0）处，旋转，再将该点反向移回
        | 1 0 dx |   | cosθ  -sinθ  0 |   | 1 0 -dx |
    M = | 0 1 dy | · | sinθ  cosθ   0 | · | 0 1 -dy |
        | 0 0 1  |   | 0     0      1 |   | 0 0  1  |
    '''
    src = cv2.imread("showimage/gongfu.jpg", cv2.IMREAD_COLOR)
    θ = 45
    '''
    scale 为缩放比例
    angle 为度数，而不是弧度
    注意是逆时针旋转
    '''
    mt = cv2.getRotationMatrix2D(
        (src.shape[1]/2, src.shape[0]/2), θ, 1)  # 以图像中心点旋转

    dst = cv2.warpAffine(src, mt, (src.shape[1], src.shape[0]),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    util.showMultipleImage((1, 2), [
        {"title": "src", "image": cv2.cvtColor(
            src, cv2.COLOR_BGR2RGB), "cmap": ''},
        {"title": "Translational", "image": cv2.cvtColor(
            dst, cv2.COLOR_BGR2RGB), "cmap": ''},
    ])


def run26():
    '''
    图像翻转
    '''
    src = cv2.imread("showimage/gongfu.jpg", cv2.IMREAD_COLOR)

    # horizontal(0), vertical(1), or both axes(-1)
    dst1 = cv2.flip(src, 0)
    dst2 = cv2.flip(src, 1)
    dst3 = cv2.flip(src, -1)

    util.showMultipleImage((1, 4), [
        {"title": "src", "image": cv2.cvtColor(
            src, cv2.COLOR_BGR2RGB), "cmap": ''},
        {"title": "dst1", "image": cv2.cvtColor(
            dst1, cv2.COLOR_BGR2RGB), "cmap": ''},
        {"title": "dst2", "image": cv2.cvtColor(
            dst2, cv2.COLOR_BGR2RGB), "cmap": ''},
        {"title": "dst2", "image": cv2.cvtColor(
            dst3, cv2.COLOR_BGR2RGB), "cmap": ''},
    ])


def run27():
    '''
    图像金字塔
    '''
    src = cv2.imread("showimage/girl.jpg", cv2.IMREAD_COLOR)  # // 800 * 732
    for i in range(3):
        dst = cv2.pyrDown(src)
        cv2.imshow("dst"+str(i), dst)
        src = dst.copy()

    cv2.waitKey(0)


def run28():
    '''
    错切，斜切，扭变
    '''
    src = cv2.imread("showimage/box.jpg", cv2.IMREAD_COLOR)
    θ = math.pi / 12
    m = np.float32([
        [1, math.tan(θ), 0],
        [0, 1, 0],
    ])
    dst = cv2.warpAffine(src, m, (src.shape[1], src.shape[0]),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    util.showMultipleImage((1, 2), [
        {"title": "src", "image": cv2.cvtColor(
            src, cv2.COLOR_BGR2RGB), "cmap": ''},
        {"title": "Translational", "image": cv2.cvtColor(
            dst, cv2.COLOR_BGR2RGB), "cmap": ''},
    ])


def run29():
    '''
    投影变换
    Perspective 透视投影
    '''
    src = cv2.imread("showimage/box.jpg", cv2.IMREAD_COLOR)
    h, w = src.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts2 = np.float32([[0, 0], [w, 0], [w-50, h-50], [50, h-50]])
    m = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(src, m, (src.shape[1], src.shape[0]))

    util.showMultipleImage((1, 2), [
        {"title": "src", "image": cv2.cvtColor(
            src, cv2.COLOR_BGR2RGB), "cmap": ''},
        {"title": "dst", "image": cv2.cvtColor(
            dst, cv2.COLOR_BGR2RGB), "cmap": ''},
    ])


def run30():
    '''
    投影变换
    '''
    src = cv2.imread("showimage/box.jpg", cv2.IMREAD_COLOR)
    h, w = src.shape[:2]
    # 根据图像中不共线的四个点在变换前后的对应位置求得3*3的变换矩阵
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts2 = np.float32([[0, 0], [w/2, 0], [w/2-50, h/2-50], [50, h/2-50]])
    m = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(src, m, (src.shape[1], src.shape[0]))

    util.showMultipleImage((1, 2), [
        {"title": "src", "image": cv2.cvtColor(
            src, cv2.COLOR_BGR2RGB), "cmap": ''},
        {"title": "dst", "image": cv2.cvtColor(
            dst, cv2.COLOR_BGR2RGB), "cmap": ''},
    ])


def run31():
    '''
    直角坐标（笛卡尔坐标）与极坐标的转换
    '''

    x = np.float32([0, 1, 2, 0, 1, 2, 0, 1, 2]) - 1
    y = np.float32([0, 0, 0, 1, 1, 1, 2, 2, 2]) - 1
    n = np.arange(9)

    '''
    x, y：直角坐标系的横坐标、纵坐标
        magnitude, angle：极坐标系的向量值、角度值
        angleInDegrees：弧度制/角度值选项，默认值 0 选择弧度制，1 选择角度制（[0,360]）
        直接坐标 --> 极坐标
    '''
    r, theta = cv2.cartToPolar(x, y, angleInDegrees=True)

    # 极坐标 --> 直接坐标
    xr, yr = cv2.polarToCart(r, theta, angleInDegrees=1)
    print(xr, yr)


def run32():
    '''
    直角坐标（笛卡尔坐标）与极坐标的转换
    '''
    img = cv2.imread("showimage/circle.jpg", cv2.IMREAD_COLOR)
    h, w = img.shape[:2]  # 图片的高度和宽度
    cx, cy = int(w/2), int(h/2)  # 以图像中心点作为变换中心
    maxR = max(cx, cy)  # 最大变换半径
    imgPolar = cv2.linearPolar(img, (cx, cy), maxR, cv2.INTER_LINEAR)
    imgPR = cv2.rotate(imgPolar, cv2.ROTATE_90_COUNTERCLOCKWISE)

    util.showMultipleImage((1, 2), [
        {"title": "img", "image": cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB), "cmap": ''},
        {"title": "imgPR", "image": cv2.cvtColor(
            imgPR, cv2.COLOR_BGR2RGB), "cmap": ''},
    ])


def run33():
    '''
    比特平面分层
    '''
    img = cv2.imread("showimage/cat.jpg", cv2.IMREAD_COLOR)
    height, width = img.shape[:2]  # 图片的高度和宽度

    plt.figure(figsize=(10, 8))
    for l in range(9, 0, -1):
        plt.subplot(3, 3, (9-l)+1, xticks=[], yticks=[])
        if l == 9:
            plt.imshow(img, cmap='gray'), plt.title('Original')
        else:
            imgBit = np.empty((height, width), dtype=np.uint8)  # 创建空数组
            for w in range(width):
                for h in range(height):
                    # 以字符串形式返回输入数字的二进制表示形式
                    x = np.binary_repr(img[h, w], width=8)
                    x = x[::-1]
                    a = x[l-1]
                    imgBit[h, w] = int(a)  # 第 i 位二进制的值
            plt.imshow(imgBit, cmap='gray')
            plt.title(f"{bin((l-1))}")
    plt.show()


def run34():
    '''
    灰度直方图
    '''
    img = cv2.imread("showimage/cat.jpg", cv2.IMREAD_GRAYSCALE)
    histCV = cv2.calcHist([img], [0], None, [256], [0, 256])
    histNP, bins = np.histogram(img.flatten(), 256)
    print(histCV.shape, histNP.shape)

    plt.figure(figsize=(10, 3))
    plt.subplot(131), plt.imshow(img, cmap='gray', vmin=0,
                                 vmax=255), plt.title("Original"), plt.axis('off')
    plt.subplot(132, xticks=[], yticks=[]), plt.axis(
        [0, 255, 0, np.max(histCV)])
    plt.bar(range(256), histCV[:, 0]), plt.title("Gray Hist(cv2.calcHist)")
    plt.subplot(133, xticks=[], yticks=[]), plt.axis(
        [0, 255, 0, np.max(histCV)])
    plt.bar(bins[:-1], histNP), plt.title("Gray Hist(np.histogram)")
    plt.show()


def run35():
    '''
    直方图均衡化
    '''
    img = cv2.imread("showimage/cat.jpg", cv2.IMREAD_GRAYSCALE)
    dst = cv2.equalizeHist(img)
    util.showMultipleImage((1, 2), [
        {"title": "img", "image": cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB), "cmap": ''},
        {"title": "dst", "image": cv2.cvtColor(
            dst, cv2.COLOR_BGR2RGB), "cmap": ''},
    ])


def run36():
    '''
    反色变换
    '''
    img = cv2.imread("showimage/cat.jpg", cv2.IMREAD_GRAYSCALE)
    dst = 255 - img
    util.showMultipleImage((1, 2), [
        {"title": "img", "image": cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB), "cmap": ''},
        {"title": "dst", "image": cv2.cvtColor(
            dst, cv2.COLOR_BGR2RGB), "cmap": ''},
    ])

def run37():
    help(cv2.IMREAD_COLOR)

def run():
    run37()

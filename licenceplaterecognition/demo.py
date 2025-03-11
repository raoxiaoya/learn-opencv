import cv2
import numpy as np

def run():
    src = cv2.imread('car_band.jpeg', cv2.IMREAD_COLOR) # 600*400

    # 规范大小
    resize = cv2.resize(src, (620, 480), None, 0, 0, cv2.INTER_LINEAR)

    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

    # 使用双边滤波过滤掉不需要的细节
    filtered = cv2.bilateralFilter(gray, 13, 15, 15)

    # 边缘检测
    # 仅显示强度梯度大于最小阈值且小于最大阈值的边缘
    cannyed = cv2.Canny(filtered, 30, 200)

    # 寻找轮廓
    contours, _ = cv2.findContours(cannyed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL, RETR_TREE
    # print(type(contours))
    # print(type(contours[0][0][0][0]))

    # 输出重定向到文件
    # with open('output.txt', 'w') as f:
    #     print(originPointsVector, file=f)

    # 轮廓画出来，-1 表示画出所有轮廓
    shape = resize.shape
    mat = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    cv2.drawContours(mat, contours, -1, (0, 255, 0), 1)

    # 求得每一个轮廓的面积
    areas = []
    areasOld = []
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)
        areasOld.append(area)

    # 按面积排序，从小到大
    areas.sort(reverse=True)
    # 提取出面积最大的10个轮廓
    areas = areas[:10]

    # 找到面积最大的10个轮廓的索引
    newContours = []
    for area in areas:
        for k, v in enumerate(areasOld):
            if areasOld[k] == area:
                newContours.append(contours[k])

    newMat = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    cv2.drawContours(newMat, newContours, -1, (0, 255, 0), 1)

    cv2.imshow('mat', mat)
    cv2.imshow('newMat', newMat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run2():
    # 创建一个空白图像
    image = np.zeros((500, 500, 3), dtype=np.uint8)

    # 创建一个简单的形状（矩形）
    cv2.rectangle(image, (100, 100), (400, 400), (255, 255, 255), 2)

    # 转换为灰度图像并二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制所有轮廓
    mat = np.zeros((600, 600, 3), dtype=np.uint8)
    cv2.drawContours(mat, contours, -1, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Contours", mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

run()
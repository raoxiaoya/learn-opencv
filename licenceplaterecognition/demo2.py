import cv2
import numpy as np
from PIL import Image
import pytesseract

# pip install opencv-python pytesseract Pillow

# 如果 Tesseract 没有添加到环境变量，请设置其路径
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用高斯模糊来减少噪声
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 使用Canny边缘检测
    edged = cv2.Canny(blur, 50, 150)
    return edged

def find_plate_contours(edged):
    # 寻找轮廓
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 根据面积排序并获取前30个
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    plate_contour = None

    for contour in contours:
        # 计算轮廓的边界框
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        # 如果我们的近似轮廓有4个点，则假设我们找到了车牌
        if len(approx) == 4:
            plate_contour = approx
            break

    return plate_contour

def recognize_plate(image, plate_contour):
    if plate_contour is not None:
        # 提取车牌区域
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate_region = image[y:y+h, x:x+w]
        # 将车牌区域转换为灰度图
        plate_gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        # 使用阈值化提高字符识别率
        _, plate_thresh = cv2.threshold(plate_gray, 150, 255, cv2.THRESH_BINARY)
        
        # 使用PyTesseract识别字符
        text = pytesseract.image_to_string(Image.fromarray(plate_thresh), lang='chi_sim+eng', config='--psm 11') # 根据需要选择语言
        print("Detected License Plate Number is:", text.strip())
        cv2.imshow("Plate", plate_thresh)
    else:
        print("No license plate found")

if __name__ == "__main__":
    # 加载图片
    image_path = "car_band.jpeg"
    image = cv2.imread(image_path)
    
    # 图像预处理
    edged = preprocess_image(image)
    
    # 查找车牌轮廓
    plate_contour = find_plate_contours(edged)
    
    # 识别车牌
    recognize_plate(image, plate_contour)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
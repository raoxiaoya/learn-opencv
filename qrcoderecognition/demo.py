import cv2
from pyzbar import pyzbar
import numpy as np

def decode_qr_code(image):
    # 使用 pyzbar 解码图像中的所有二维码
    decoded_objects = pyzbar.decode(image)
    
    for obj in decoded_objects:
        # 打印二维码中的数据
        print("Type:", obj.type)  # 类型（例如 QR_CODE）
        print("Data: ", obj.data.decode("utf-8"), "\n")  # 数据内容
        
        # 获取二维码的位置，并在图像上绘制矩形框
        points = obj.polygon
        
        # 如果二维码是四边形，则绘制四个点
        if len(points) == 4:
            pts = [(point.x, point.y) for point in points]
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    return image

if __name__ == "__main__":
    # 加载图像
    image_path = "4.jpg"
    image = cv2.imread(image_path)
    
    # 解码二维码并获取带有标注的图像
    output_image = decode_qr_code(image)
    
    # 显示结果图像
    cv2.imshow("QR Code Detection", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
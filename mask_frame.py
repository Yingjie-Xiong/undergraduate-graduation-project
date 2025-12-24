import cv2
import numpy as np

# 读取输入图和mask图
input_image = cv2.imread('C:/Users\86185\Desktop/1/image01497.jpg')
mask_image = cv2.imread('C:/Users\86185\Desktop/1/image00001_crop000_mask000.png', cv2.IMREAD_GRAYSCALE)

# 白色部分覆盖原图 (白色部分保留全白图内容)
white_overlay = np.ones_like(input_image) * 255  # 创建全白的RGB图像
overlay_area = cv2.bitwise_and(white_overlay, white_overlay, mask=mask_image)

# 黑色部分保留原图内容 (黑色部分直接保留原图)
masked_area = cv2.bitwise_and(input_image, input_image, mask=cv2.bitwise_not(mask_image))

# 合并两个区域
result = cv2.add(masked_area, overlay_area)


# 保存并显示结果

cv2.imwrite('C:/Users\86185\Desktop/1/result_image1.jpg', result)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


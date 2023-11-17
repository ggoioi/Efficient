import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt

# TIFF 이미지 불러오기
image = tifffile.imread('C:/Users/piai/Desktop/EfficientAD/Data_test/test/class/1_1불량.tiff')

# 만약 이미지가 컬러라면 그레이스케일로 변환
# print(image.ndim)
# if image.ndim == 3:
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# 이미지 데이터 타입을 uint8로 변환
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 이미지 이진화 (binary thresholding)
ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 외곽선 찾기
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 원본 이미지 복사 후 외곽선 그리기
image_contour = np.copy(image)
cv2.drawContours(image_contour, contours, -1, (0, 255, 0), 2)

# 결과 표시
plt.imshow(image_contour, cmap='gray')
plt.show()
# 이미지를 뭘로 쓰냐가 중요함
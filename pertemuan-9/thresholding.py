import cv2
import numpy as np

# Baca citra grayscale
image = cv2.imread('assets\img\DSC01097.JPG', cv2.IMREAD_GRAYSCALE)

resize = cv2.resize(image, (400, 400))

# Global thresholding
ret, thresh1 = cv2.threshold(resize, 127, 255, cv2.THRESH_BINARY)

# Adaptive thresholding
thresh2 = cv2.adaptiveThreshold(
    resize, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# Tampilkan hasil
cv2.imshow('Global Thresholding', thresh1)
cv2.imshow('Adaptive Thresholding', thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()

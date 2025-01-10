import cv2
import numpy as np

# Baca citra dalam grayscale
image = cv2.imread('assets\img\e.JPG', cv2.IMREAD_GRAYSCALE)

# Terapkan thresholding global
ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Tampilkan hasil segmentasi
cv2.imshow('Thresholded Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
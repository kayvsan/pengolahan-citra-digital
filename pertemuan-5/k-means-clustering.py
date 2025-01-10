import cv2
import numpy as np

# Membaca gambar
image = cv2.imread('assets\img\e.JPG')
Z = image.reshape((-1, 3))

# Konversi ke float32
Z = np.float32(Z)

# Kriteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Kembalikan ke uint8 dan reshape
center = np.uint8(center)
res = center[label.flatten()]
segmented_image = res.reshape((image.shape))

# Menampilkan hasil
cv2.imshow('K-means Segmentation', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

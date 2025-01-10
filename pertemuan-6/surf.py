import cv2

# Membaca gambar
image = cv2.imread('assets\img\e.JPG')

# Inisialisasi objek ORB
orb = cv2.ORB_create()

# Mendeteksi keypoints dan komputasi deskriptor
keypoints, descriptors = orb.detectAndCompute(image, None)

# Menggambar keypoints di citra
orb_image = cv2.drawKeypoints(image, keypoints, None)

# Menampilkan hasil
cv2.imshow('ORB Features', orb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

# Muat file Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Baca citra
image = cv2.imread('assets\img\e.JPG')
resize = cv2.resize(image, (500, 800))
gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

# Deteksi wajah
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Gambar bounding box di sekitar wajah yang terdeteksi
for (x, y, w, h) in faces:
    cv2.rectangle(resize, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Tampilkan hasil
cv2.imshow('Face Detection', resize)
cv2.waitKey(0)
cv2.destroyAllWindows()

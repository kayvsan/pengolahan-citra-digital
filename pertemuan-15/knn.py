import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Muat dataset MNIST
mnist = fetch_openml('mnist_784', version=1)

# Ambil fitur (X) dan label (y)
X = mnist.data
y = mnist.target.astype(int) 

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Latih model pada data training
knn.fit(X_train, y_train)

# Prediksi pada data testing
y_pred = knn.predict(X_test)

# Hitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model KNN pada dataset MNIST: {accuracy * 100:.2f}%")

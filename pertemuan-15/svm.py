from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Memuat dataset Iris
iris = datasets.load_iris()
X = iris.data  # Fitur (panjang & lebar kelopak/daun)
y = iris.target  # Label (setosa, versicolor, virginica)

# 2. Membagi dataset menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Melatih model SVM dengan kernel linear
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# 4. Memprediksi data uji
y_pred = model.predict(X_test)

# 5. Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi SVM pada Iris Dataset: {accuracy * 100:.2f}%")


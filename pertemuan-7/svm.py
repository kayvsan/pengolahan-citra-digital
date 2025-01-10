import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'assets\dataset\cars_dataset.csv'  
cars_dataset = pd.read_csv(file_path)

# Encode categorical features into numeric values
label_encoders = {}
encoded_data = cars_dataset.copy()

# Apply LabelEncoder to all columns
for column in encoded_data.columns:
    le = LabelEncoder()
    encoded_data[column] = le.fit_transform(encoded_data[column])
    label_encoders[column] = le

# Split features (X) and target (y)
X = encoded_data.drop(columns=['car']).values
y = encoded_data['car'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize SVM model with a linear kernel
clf = svm.SVC(kernel='linear')

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy * 100:.2f}%')

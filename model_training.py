import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

csv_file = 'C:/Users/mansi/Documents/mini/processed/final/features.csv'
features = pd.read_csv(csv_file)

csv_file = 'labeled_audio_data.csv'
labels = pd.read_csv(csv_file)

features = np.random.rand(3200, 13)
labels = np.random.randint(0, 7, 3200)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
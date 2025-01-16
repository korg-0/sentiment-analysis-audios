import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

features_file = 'C:/Users/mansi/Documents/mini/final/features.csv'
labels_file = 'C:/Users/mansi/Documents/mini/labeled_audio_data.csv'

features = pd.read_csv(features_file) 
labels = pd.read_csv(labels_file)      

X = features.iloc[:, 1:].values  
y = labels.iloc[:, 1].values    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='rbf', C=5.0, gamma='scale')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred,zero_division=0))
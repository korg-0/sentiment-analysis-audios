import os
from sklearn.preprocessing import LabelEncoder

input_dir = 'C:/Users/mansi/Documents/mini/processed/final'

file_names = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
labels = [file_name.split('.')[2].split('-')[0] for file_name in file_names]

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print("Label Mapping:", label_mapping)

labeled_data = [{'file_name': file_name, 'label': label} for file_name, label in zip(file_names, encoded_labels)]

import pandas as pd

df = pd.DataFrame(labeled_data)
df.to_csv('labeled_audio_data.csv', index=False)
print("Labeled data saved to labeled_audio_data.csv")

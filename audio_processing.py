import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd

input_dir = 'C:/Users/mansi/Documents/mini/processed'
output_dir = 'C:/Users/mansi/Documents/mini/final'
os.makedirs(output_dir, exist_ok=True)

def process_audio(file_path, target_sr=16000, duration=5):
    try:
        y, sr = librosa.load(file_path, sr=None)  
        y = librosa.util.normalize(y)
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr) 
        y, _ = librosa.effects.trim(y, top_db=20)
        fixed_length = target_sr * duration
        if len(y) < fixed_length:
            y = librosa.util.fix_length(y, size=fixed_length)
        else:
            y = y[:fixed_length]
        mfccs = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=13)
        mfccs_mean = mfccs.mean(axis=1) 
        return y, target_sr, mfccs_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

features = []
j=0
for file_name in os.listdir(input_dir):
    if file_name.endswith('.wav'):  
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing: {file_name}")
        y_processed, sr_processed, mfccs_mean = process_audio(file_path)
        if y_processed is not None:
            output_path = os.path.join(output_dir, file_name)
            sf.write(output_path, y_processed, sr_processed)
            print(f"Saved processed file: {output_path}")
            feature_row = {'file_name': j}
            for i, mfcc_value in enumerate(mfccs_mean):
                feature_row[f'mfcc_{i+1}'] = mfcc_value  
            features.append(feature_row)
    j+=1

features_df = pd.DataFrame(features)
csv_path = os.path.join(output_dir, 'features.csv')
features_df.to_csv(csv_path, index=False)
print(f"Feature extraction completed and saved to {csv_path}")
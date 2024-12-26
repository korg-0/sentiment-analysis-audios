import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd

input_dir = 'C:/Users/mansi/Documents/mini/processed'  
output_dir = 'C:/Users/mansi/Documents/mini/processed/final'  
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
        mfccs = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=13).mean(axis=1)
        return y, target_sr, mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

features = []
for file_name in os.listdir(input_dir):
    if file_name.endswith('.wav'):  
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing: {file_name}")

        y_processed, sr_processed, mfccs = process_audio(file_path)
        if y_processed is not None:
            output_path = os.path.join(output_dir, file_name)
            sf.write(output_path, y_processed, sr_processed)
            print(f"Saved processed file: {output_path}")
            features.append({'file_name': file_name, 'mfccs': mfccs})

features_df = pd.DataFrame(features)
features_df.to_csv(os.path.join(output_dir, 'features.csv'), index=False)
print("Feature extraction completed and saved to features.csv")

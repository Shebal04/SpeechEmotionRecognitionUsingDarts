import os
import warnings
import numpy as np
import librosa

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Extract MFCC features from a single audio file
def extract_features(file_path, max_pad_len=None):
    try:
        # Load audio file
        print(f"Loading audio file: {file_path}")  # Debug line
        audio, sample_rate = librosa.load(file_path, sr=None, res_type='kaiser_fast')
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        print(f"MFCCs computed for {file_path}, shape: {mfccs.shape}")  # Debug line
        
        # Transpose to have time steps first
        mfccs = mfccs.T
        
        # Pad or truncate MFCCs to fixed length
        if max_pad_len is not None:
            if mfccs.shape[0] > max_pad_len:
                print(f"Truncating MFCCs to {max_pad_len} time steps")  # Debug line
                mfccs = mfccs[:max_pad_len, :]
            else:
                pad_width = max_pad_len - mfccs.shape[0]
                print(f"Padding MFCCs to {max_pad_len} time steps")  # Debug line
                mfccs = np.pad(mfccs, pad_width=((0, pad_width), (0, 0)), mode='constant')

        return mfccs

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process a full dataset directory
def process_dataset(data_paths, max_pad_len=None):
    features = []
    labels = []
    file_paths = []
    emotion_counts = {}

    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    
    for data_path in data_paths:
        print(f"Processing directory: {data_path}")

        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    print(f"Processing file: {file_path}")
                    
                    # Parse emotion from filename
                    parts = file.split('-')
                    if len(parts) >= 7:
                        emotion_code = parts[2]
                        emotion = emotion_map.get(emotion_code, 'unknown')
                    else:
                        emotion = 'unknown'

                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                    # Extract MFCC features
                    feature = extract_features(file_path, max_pad_len)
                    if feature is not None:
                        features.append(feature)
                        labels.append(emotion)
                        file_paths.append(file_path)

    # Print class distribution
    print("\nEmotion Distribution:")
    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count} samples")

    return np.array(features), np.array(labels), file_paths

# Run as a standalone script
if __name__ == "__main__":
    # Paths where dataset folders are located
    data_paths = [
        "D:/Ser/data/audio/Audio_Song_Actors_01-24",
        "D:/Ser/data/audio/Audio_Speech_Actors_01-24"
    ]

    print("Processing dataset...")  # Debug line
    features, labels, file_paths = process_dataset(data_paths, max_pad_len=100)
    
    print(f"\nProcessed {len(features)} files successfully.")
    print(f"Features shape: {features.shape}")
    print(f"Unique emotions found: {np.unique(labels)}")

    # Display class balance
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution summary:")
    for label, count in zip(unique_labels, counts):
        print(f"{label}: {count} samples ({(count/len(labels))*100:.2f}%)")

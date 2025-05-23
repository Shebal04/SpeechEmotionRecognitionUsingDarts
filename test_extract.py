import os
import numpy as np
from preprocess import extract_features, process_dataset

def test_single_file(file_path):
    """Test feature extraction on a single audio file."""
    print(f"Testing extraction on single file: {file_path}")
    features = extract_features(file_path)
    
    if features is not None:
        print(f"Success! Features shape: {features.shape}")
        # Optionally visualize the features
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 4))
        # plt.imshow(features, aspect='auto', origin='lower')
        # plt.colorbar()
        # plt.title('MFCC Features')
        # plt.tight_layout()
        # plt.show()
    else:
        print(f"Failed to extract features from {file_path}")

def test_small_batch(data_path, max_files=5):
    """Test feature extraction on a small batch of files."""
    print(f"Testing extraction on up to {max_files} files from {data_path}")
    
    count = 0
    success = 0
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav') and count < max_files:
                file_path = os.path.join(root, file)
                print(f"\nProcessing: {file_path}")
                
                features = extract_features(file_path)
                count += 1
                
                if features is not None:
                    success += 1
                    print(f"Success! Features shape: {features.shape}")
                else:
                    print(f"Failed to extract features.")
                    
    print(f"\nProcessed {count} files with {success} successes and {count-success} failures.")

def test_full_dataset_processing(data_path, max_pad_len=100):
    """Test processing the entire dataset with standardized length."""
    print(f"Testing full dataset processing from {data_path}")
    
    try:
        features, labels, file_paths = process_dataset(data_path, max_pad_len)
        print(f"Successfully processed {len(features)} files")
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Print some statistics about the dataset
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\nEmotion distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count} samples")
            
    except Exception as e:
        print(f"Error in dataset processing: {str(e)}")

if __name__ == "__main__":
    # Path to your dataset
    data_path = "D:/Ser/data/audio/Audio_Song_Actors_01-24"
    
    # 1. Test a specific file that was causing problems
    problem_file = "D:/Ser/data/audio/Audio_Song_Actors_01-24/Actor_01/03-02-01-01-01-01.wav"
    if os.path.exists(problem_file):
        test_single_file(problem_file)
    else:
        print(f"Problem file does not exist: {problem_file}")
        # Try to find the actual file
        actor_dir = os.path.dirname(problem_file)
        if os.path.exists(actor_dir):
            print("Files in the actor directory:")
            for f in os.listdir(actor_dir):
                if f.endswith('.wav'):
                    print(f"  {f}")
            # Try the first available wav file
            wav_files = [f for f in os.listdir(actor_dir) if f.endswith('.wav')]
            if wav_files:
                alt_file = os.path.join(actor_dir, wav_files[0])
                print(f"\nTrying alternative file: {alt_file}")
                test_single_file(alt_file)
    
    print("\n" + "-"*50 + "\n")
    
    # 2. Test a small batch of files
    test_small_batch(data_path, max_files=3)
    
    print("\n" + "-"*50 + "\n")
    
    # 3. Test processing the entire dataset (uncomment when ready)
    # test_full_dataset_processing(data_path)
    
    print("\nFeature extraction testing complete!")
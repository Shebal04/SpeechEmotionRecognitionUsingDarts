import torch
import numpy as np
import os
from final_model.model_final import Network
from search.genotypes import load_genotype_from_file
from preprocess import extract_features  # <-- import from your preprocess.py

# Testing function for a single audio file
def test_audio(file_path, model, max_pad_len=100):
    # Emotion label mapping
    emotion_map = {
        'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3, 
        'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
    }
    inverse_emotion_map = {v: k for k, v in emotion_map.items()}

    # Extract features using your preprocess.py
    features = extract_features(file_path, max_pad_len=max_pad_len)
    
    if features is None:
        print(f"Failed to extract features from {file_path}")
        return

    # Prepare tensor for model input
    input_tensor = torch.FloatTensor(features)
    input_tensor = input_tensor.unsqueeze(0)  # Batch dimension
    input_tensor = input_tensor.unsqueeze(0)  # Channel dimension
    # Final shape: [1, 1, time_steps, n_mfcc]

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Get prediction
    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()

    print(f"\nPredicted emotion: {inverse_emotion_map[prediction]}")
    print(f"Probabilities: {probabilities[0].numpy()}")
    
    return prediction

# Function to load the model
def load_model(model_path='final_model.pt'):
    device = torch.device('cpu')  # Using CPU for testing

    # Load the best architecture (genotype)
    if os.path.exists('best_genotype.txt'):
        genotype = load_genotype_from_file('best_genotype.txt')
    else:
        print("No genotype file found. Using default architecture.")
        genotype = None

    # Initialize model
    model = Network(C=32, num_classes=8, layers=6, genotype=genotype).to(device)

    # Load trained model weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file {model_path} not found. Using random weights.")

    model.eval()  # Evaluation mode
    return model

# Main execution
if __name__ == "__main__":
    # Load model
    model = load_model()

    # Path to your test audio file (change it here)
    file_path = r"D:/Ser/sad.wav"  # <-- Update with your own test file path

    print(f"\nTesting audio file: {file_path}")
    test_audio(file_path, model)

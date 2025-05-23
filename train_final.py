import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import numpy as np

from final_model.model_final import Network  # Import your final model
from search.genotypes import load_genotype_from_file  # Load the final genotype
from preprocess import extract_features  # Import your feature extraction function

# Define the real SER dataset
class SERDataset(Dataset):
    def __init__(self, data_paths, max_pad_len=100):
        self.features = []
        self.labels = []
        self.label_map = {
            'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
            'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
        }
        
        print("Loading dataset...")  # Debug line
        for data_path in data_paths:
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.wav'):
                        file_path = os.path.join(root, file)
                        parts = file.split('-')
                        if len(parts) >= 7:
                            emotion_code = parts[2]
                            emotion_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                                           '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
                            emotion = emotion_map.get(emotion_code)
                            
                            if emotion is not None:
                                print(f"Extracting feature for {file_path}...")  # Debug line
                                feature = extract_features(file_path, max_pad_len=max_pad_len)
                                if feature is not None:
                                    self.features.append(feature)
                                    self.labels.append(self.label_map[emotion])

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        print(f"Dataset loaded: {len(self.features)} samples.")  # Debug line

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)  # Shape [1, height, width]
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return feature_tensor, label_tensor


# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    print("Starting training loop...")  # Debug line
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# Main training pipeline
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")  # Debug line
    
    # Load genotype
    if os.path.exists('best_genotype.txt'):
        print("Loading genotype...")  # Debug line
        genotype = load_genotype_from_file('best_genotype.txt')
    else:
        print("No genotype file found. Proceeding without a fixed architecture.")  # Debug line
        genotype = None

    # Define model
    num_classes = 8
    model = Network(C=32, num_classes=num_classes, layers=6, genotype=genotype).to(device)

    # Training settings
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 50

    # Load dataset
    data_paths = [
        "D:/Ser/data/audio/Audio_Song_Actors_01-24",
        "D:/Ser/data/audio/Audio_Speech_Actors_01-24"
    ]
    print(f"Loading dataset from {data_paths}")  # Debug line
    train_dataset = SERDataset(data_paths, max_pad_len=100)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"Loaded {len(train_dataset)} training samples.")  # Debug line

    # Define loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} begins...")  # Debug line
        avg_loss, accuracy = train(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")  # Debug line

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"final_checkpoint_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")  # Debug line

    # Save final model
    torch.save(model.state_dict(), 'final_model.pt')
    print("Training complete. Final model saved as 'final_model.pt'.")  # Debug line


if __name__ == '__main__':
    main()

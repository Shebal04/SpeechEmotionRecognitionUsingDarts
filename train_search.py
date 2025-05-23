import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from search.genotypes import example_genotype, save_genotype_to_file

# Create the genotype file if it doesn't exist
if not os.path.exists('best_genotype.txt'):
    print("Creating 'best_genotype.txt' with the example genotype.")
    save_genotype_to_file(example_genotype, 'best_genotype.txt')

# Now import the Network class after potentially creating the genotype file
from final_model.model_final import Network  # Import the final model

# Define a dataset for the final task (replace this with your actual dataset)
class DummyDataset(Dataset):
    def __init__(self, num_samples=100, input_size=(1, 28, 28), num_classes=8):
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes
        self.data = torch.randn(num_samples, *input_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_final(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Debug info
        if batch_idx == 0:
            print(f"Input data shape: {data.shape}")
        
        optimizer.zero_grad()
        
        try:
            # Forward pass
            output = model(data)
            
            if batch_idx == 0:
                print(f"Output shape: {output.shape}")
            
            # Compute loss
            loss = criterion(output, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Get predictions and calculate accuracy
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total if total > 0 else 0
    
    return avg_loss, accuracy

# Main function to initialize and train the final model
def main():
    # Device configuration (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 50
    
    # Create DataLoader for training (Replace with your actual dataset)
    train_dataset = DummyDataset(num_samples=1000, input_size=(1, 28, 28), num_classes=8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the model
    model = Network(C=32, num_classes=8, layers=6).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Scheduler for learning rate adjustment
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        
        # Train the model
        avg_loss, accuracy = train_final(model, train_loader, optimizer, criterion, device)
        
        # Adjust learning rate
        scheduler.step()
        
        # Print training statistics
        print(f"Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
        
        # Optionally, save model checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"final_checkpoint_epoch_{epoch+1}.pt")
    
    # Save the final trained model after all epochs
    torch.save(model.state_dict(), 'final_model.pt')
    print("Final model saved as 'final_model.pt'")

if __name__ == '__main__':
    main()
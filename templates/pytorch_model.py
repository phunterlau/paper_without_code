"""
Template for model-related papers using PyTorch.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class SimpleModel(nn.Module):
    """
    A simple PyTorch model template.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the model.
        
        Args:
            input_size (int): The size of the input
            hidden_size (int): The size of the hidden layer
            output_size (int): The size of the output
        """
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): The input tensor
            
        Returns:
            torch.Tensor: The output tensor
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleDataset(Dataset):
    """
    A simple PyTorch dataset template.
    """
    def __init__(self, data, targets):
        """
        Initialize the dataset.
        
        Args:
            data (np.ndarray): The input data
            targets (np.ndarray): The target data
        """
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: The length of the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx (int): The index of the item
            
        Returns:
            tuple: The data and target
        """
        return self.data[idx], self.targets[idx]

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    """
    Train the model.
    
    Args:
        model (nn.Module): The model to train
        dataloader (DataLoader): The dataloader for the training data
        criterion: The loss function
        optimizer: The optimizer
        num_epochs (int): The number of epochs to train for
        
    Returns:
        list: The training losses
    """
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for inputs, targets in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    return losses

def evaluate_model(model, dataloader, criterion):
    """
    Evaluate the model.
    
    Args:
        model (nn.Module): The model to evaluate
        dataloader (DataLoader): The dataloader for the evaluation data
        criterion: The loss function
        
    Returns:
        float: The evaluation loss
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    
    return running_loss / len(dataloader)

def generate_sample_data(num_samples=1000, input_size=10):
    """
    Generate sample data for demonstration.
    
    Args:
        num_samples (int): The number of samples to generate
        input_size (int): The size of each input
        
    Returns:
        tuple: The input data and target data
    """
    # Generate random input data
    X = np.random.randn(num_samples, input_size)
    
    # Generate target data (a simple function of the input)
    y = np.sum(X, axis=1, keepdims=True) * 0.1
    
    return X, y

def plot_losses(losses):
    """
    Plot the training losses.
    
    Args:
        losses (list): The training losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def main():
    """
    Main function to demonstrate the implementation.
    """
    parser = argparse.ArgumentParser(description="Train a simple PyTorch model")
    parser.add_argument("--input_size", type=int, default=10, help="Size of the input")
    parser.add_argument("--hidden_size", type=int, default=50, help="Size of the hidden layer")
    parser.add_argument("--output_size", type=int, default=1, help="Size of the output")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    args = parser.parse_args()
    
    # Generate sample data
    X, y = generate_sample_data(num_samples=1000, input_size=args.input_size)
    
    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create datasets and dataloaders
    train_dataset = SimpleDataset(X_train, y_train)
    test_dataset = SimpleDataset(X_test, y_test)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create the model
    model = SimpleModel(args.input_size, args.hidden_size, args.output_size)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    losses = train_model(model, train_dataloader, criterion, optimizer, num_epochs=args.num_epochs)
    
    # Evaluate the model
    test_loss = evaluate_model(model, test_dataloader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Plot the training losses
    plot_losses(losses)

if __name__ == "__main__":
    main()

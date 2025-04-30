from image_processor import image_to_edges
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from typing import Tuple, List
import platform
if platform.system() != "Windows":
    os.environ["QT_QPA_PLATFORM"] = "xcb"  # Force XCB platform for Linux

class MarbleDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        """
        Args:
            data_path: Full path to the parquet file containing the records
            transform: Optional transform to be applied on the images
        """
        self.transform = transform
        self.records = []
        
        # Load the parquet file containing the records
        print(f"Loading parquet file from {data_path}")
        df = pd.read_parquet(data_path)
        print(f"Loaded {len(df)} records")
        
        # Process each record
        for _, row in df.iterrows():
            image_path = row['screen']
            
            if os.path.exists(image_path):
                self.records.append({
                    'image_path': image_path,
                    'forward': row['input_forward'],
                    'back': row['input_back'],
                    'left': row['input_left'],
                    'right': row['input_right'],
                    'reset': row['input_reset']
                })
            else:
                print(f"Image not found: {image_path}")
        print(f"Loaded {len(self.records)} records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        
        # Read the raw screen bytes
        with open(record['image_path'], 'rb') as f:
            screen_bytes = f.read()
        
        img = image_to_edges(screen_bytes)
        
        # Create the target tensor
        target = torch.FloatTensor([
            record['forward'],
            record['back'],
            record['left'],
            record['right'],
            record['reset']
        ])
        
        return img, target

class MarbleCNN(nn.Module):
    def __init__(self):
        super(MarbleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 9, 512)  # Adjusted for input size 320x180
        self.fc2 = nn.Linear(512, 5)  # 5 outputs for the control commands
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, 128 * 16 * 9)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                num_epochs: int,
                device: torch.device,
                test_loader: DataLoader = None) -> Tuple[List[float], List[float], List[float]]:
    """
    Train the CNN model.
    
    Args:
        model: The CNN model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on (CPU/GPU)
        test_loader: Optional DataLoader for test data
    
    Returns:
        Tuple of (training losses, test losses, test accuracies)
    """
    model.train()
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} of {num_epochs}")
        running_loss = 0.0
        total_batches = 0
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_batches += 1
            
            if i % 10 == 9:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.4f}')
                train_losses.append(running_loss / 10)
                running_loss = 0.0
        
        # Print average loss for the epoch
        epoch_avg_loss = running_loss / total_batches
        print(f'Epoch {epoch + 1} completed. Average training loss: {epoch_avg_loss:.4f}')
        
        # Evaluation phase if test_loader is provided
        if test_loader is not None:
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, targets in test_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    
                    # Convert outputs to binary predictions
                    predictions = (outputs > 0.5).float()
                    correct += (predictions == targets).sum().item()
                    total += targets.numel()
            
            avg_test_loss = test_loss / len(test_loader)
            accuracy = correct / total
            test_losses.append(avg_test_loss)
            test_accuracies.append(accuracy)
            
            print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}')
    
    return train_losses, test_losses, test_accuracies

def save_model(model: nn.Module, path: str):
    """Save the trained model."""
    torch.save(model.state_dict(), path)

def load_model(model: nn.Module, path: str):
    """Load a trained model."""
    model.load_state_dict(torch.load(path))
    return model

def predict_controls(model: nn.Module, image: np.ndarray, device: torch.device, velocity: int) -> Tuple[bool, bool, bool, bool, bool]:
    """
    Predict control commands for a given image.
    
    Args:
        model: Trained CNN model
        image: Input image (BGR format)
        device: Device to run inference on
    
    Returns:
        Tuple of (forward, back, left, right, reset) boolean values
    """
    model.eval()
    with torch.no_grad():
        # Preprocess image
        img = image_to_edges(image)
        img = img.to(device)
        
        # Get predictions
        outputs = model(img)
        predictions = (outputs > 0.5).cpu().numpy()[0]
        
        return tuple(predictions) 
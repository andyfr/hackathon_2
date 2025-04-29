import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from typing import Tuple, List

class MarbleDataset(Dataset):
    def __init__(self, csv_path: str, image_dir: str, transform=None):
        """
        Args:
            csv_path: Full path to the CSV file containing the records
            image_dir: Directory containing the screen images
            transform: Optional transform to be applied on the images
        """
        self.image_dir = image_dir
        self.transform = transform
        self.records = []
        
        # Load the CSV file containing the records
        print(f"Loading CSV file from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records")
        
        # Process each record
        for _, row in df.iterrows():
            image_path = os.path.join(self.image_dir, row['filename'])
            
            if os.path.exists(image_path):
                self.records.append({
                    'image_path': image_path,
                    'forward': row['forward'],
                    'back': False,
                    'left': row['left'],
                    'right': row['right'],
                    'reset': row['reset']
                })
            else:
                print(f"Image not found: {image_path}")
        print(f"Loaded {len(self.records)} records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        
        # Load and preprocess the image
        img = cv2.imread(record['image_path'])
        if img is None:
            raise ValueError(f"Failed to load image: {record['image_path']}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 180))  # Resize to smaller dimensions
        img = img.transpose(2, 0, 1)  # Convert to CxHxW format
        img = torch.FloatTensor(img) / 255.0  # Normalize to [0, 1]
        
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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 22 * 40, 512)  # Adjusted for input size 320x180
        self.fc2 = nn.Linear(512, 5)  # 5 outputs for the control commands
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 22 * 40)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        
        return x

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                num_epochs: int,
                device: torch.device) -> List[float]:
    """
    Train the CNN model.
    
    Args:
        model: The CNN model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on (CPU/GPU)
    
    Returns:
        List of training losses
    """
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
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
                losses.append(running_loss / 10)
                running_loss = 0.0
        
        # Print average loss for the epoch
        epoch_avg_loss = running_loss / total_batches
        print(f'Epoch {epoch + 1} completed. Average loss: {epoch_avg_loss:.4f}')
    
    return losses

def save_model(model: nn.Module, path: str):
    """Save the trained model."""
    torch.save(model.state_dict(), path)

def load_model(model: nn.Module, path: str):
    """Load a trained model."""
    model.load_state_dict(torch.load(path))
    return model

def predict_controls(model: nn.Module, image: np.ndarray, device: torch.device) -> Tuple[bool, bool, bool, bool, bool]:
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
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 180))
        img = img.transpose(2, 0, 1)
        img = torch.FloatTensor(img).unsqueeze(0) / 255.0
        img = img.to(device)
        
        # Get predictions
        outputs = model(img)
        predictions = (outputs > 0.5).cpu().numpy()[0]
        
        return tuple(predictions) 
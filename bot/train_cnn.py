import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from cnn_model import MarbleCNN, MarbleDataset, train_model, save_model
import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Train CNN model for marble game')
    parser.add_argument('--data-path', type=str, default='marble_client_records_0.parquet', help='Path to the parquet file containing the records')
    parser.add_argument('--model-path', type=str, default='marble_cnn.pth', help='Path to save the trained model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test-split', type=float, default=0.2, help='Fraction of data to use for testing')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create dataset
    dataset = MarbleDataset(args.data_path)
    
    # Split dataset into train and test sets
    test_size = int(len(dataset) * args.test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f'Training set size: {train_size}')
    print(f'Test set size: {test_size}')

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model
    model = MarbleCNN().to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    print('Starting training...')
    train_losses, test_losses, test_accuracies = train_model(
        model, train_loader, criterion, optimizer, args.epochs, device, test_loader
    )

    # Save the model
    save_model(model, args.model_path)
    print(f'Model saved to {args.model_path}')
    
    # Print final metrics
    print('\nFinal Results:')
    print(f'Final Training Loss: {train_losses[-1]:.4f}')
    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    print(f'Final Test Accuracy: {test_accuracies[-1]:.4f}')

if __name__ == '__main__':
    main() 
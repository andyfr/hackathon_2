import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cnn_model import MarbleCNN, MarbleDataset, train_model, save_model
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train CNN model for marble game')
    parser.add_argument('--csv-path', type=str, default='annotations.csv', help='Path to the CSV file containing the records')
    parser.add_argument('--image-dir', type=str, default='simplified_images', help='Directory containing the screen images')
    parser.add_argument('--model-path', type=str, default='marble_cnn.pth', help='Path to save the trained model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create dataset and dataloader
    dataset = MarbleDataset(args.csv_path, args.image_dir)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Create model
    model = MarbleCNN().to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    print('Starting training...')
    losses = train_model(model, train_loader, criterion, optimizer, args.epochs, device)

    # Save the trained model
    save_model(model, args.model_path)
    print(f'Model saved to {args.model_path}')

if __name__ == '__main__':
    main() 
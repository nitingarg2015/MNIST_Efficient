import torch
import torch.nn as nn
import torch.optim as optim
from model import MNISTModel
from datetime import datetime
import os
from utils import get_model_accuracy, get_mnist_loader

def train():
    # Set device
    device = torch.device("cpu")
    
    # Load MNIST dataset using the new function
    train_loader = get_mnist_loader(train = True)
    
    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    
    running_loss = 0.0
    correct = 0
    total = 0
    # Train for one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, train_accuracy: {correct/total:.2%}')
    
    final_train_accuracy = 100 * correct / total

    test_accuracy = get_model_accuracy(model, train = False)
    print(f'test accuracy: {test_accuracy}%')
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
        # Save model with timestamp in models directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join('models', f'model_mnist_{timestamp}.pth')
    # torch.save(model.state_dict(), save_path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_accuracy': final_train_accuracy
    }, save_path)
    return save_path

if __name__ == "__main__":
    train() 
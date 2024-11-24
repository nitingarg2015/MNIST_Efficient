import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

class TrainingManager:
    def __init__(self, model):
        self.device = torch.device('cpu')
        self.model = model.to(self.device)
        # Data loading
        self.train_loader, self.test_loader = self.load_data()
        # initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        self.training_accuracy = 0
        self.test_accuracy = 0
        
    def load_data(self):
        
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation(degrees=5),      # Slight rotation (reduced from 15 to 10 degrees)
                transforms.RandomAffine(
                degrees=0,                             # No rotation in affine transform
                translate=(0.05, 0.05),                  # Random translation up to 10%
                scale=(0.95, 1.05)                       # Random scaling between 90% and 110%
            ),
            transforms.GaussianBlur(
                kernel_size=3,                         # Blur kernel size
                sigma=(0.01, 0.02)                       # Random sigma for blur
                ),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            # Original transform without augmentation
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
        self.test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
        
        train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=64)

        return train_loader, test_loader
    
    def train(self, optimizer):
        train_losses = []
        train_acc = []

        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(self.device), target.to(self.device)

            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = self.model(data)

            # Calculate loss
            loss = F.nll_loss(y_pred, target)
            train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update pbar-tqdm
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            train_acc.append(100*correct/processed)
        
        self.training_accuracy = 100*correct/processed
    
    def test(self):
        test_losses = []
        test_acc = []

        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
        
        test_acc.append(100. * correct / len(self.test_loader.dataset))
        self.test_accuracy = 100. * correct / len(self.test_loader.dataset)
        return self.test_accuracy, test_loss

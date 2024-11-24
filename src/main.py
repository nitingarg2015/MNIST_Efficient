
import os
import sys
from pathlib import Path
from torch.optim.lr_scheduler import StepLR # Import your choice of scheduler here
from torch import optim
import torch
from datetime import datetime

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import MNISTModel
from src.training_manager import TrainingManager

def train_test():
    model = MNISTModel(dropout_rate=0.1)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    training_manager = TrainingManager(model)
    patience = 3
    target_test_accuracy = 99.4
    best_loss = float('inf')
    patience_counter = 0
    accuracy_counter = 0
    delta = 0.002

    for epoch in range(1):
        print("EPOCH:", epoch, "Learning Rate: ", optimizer.param_groups[0]["lr"])

        training_manager.train(optimizer)
        test_accuracy, test_loss = training_manager.test()
        scheduler.step()

        # Early stopping logic
        if test_loss < best_loss - delta:
            best_loss = test_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                model.load_state_dict(best_model_state)  # Restore best model
                break
        
        if test_accuracy > target_test_accuracy:
            accuracy_counter += 1
            if accuracy_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                model.load_state_dict(best_model_state)  # Restore best model
                break
        else:
            accuracy_counter = 0
            
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
        # Save model with timestamp in models directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join('models', f'model_mnist_{timestamp}.pth')
    # torch.save(model.state_dict(), save_path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_accuracy': training_manager.training_accuracy,
        'test_accuracy': training_manager.test_accuracy,
        'epochs': epoch
    }, save_path)

    return save_path

if __name__ == "__main__":
    train_test() 
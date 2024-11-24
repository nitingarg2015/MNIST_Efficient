import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def count_parameters(model):
    # print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters')
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 


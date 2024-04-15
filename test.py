import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork():
    def __init__(self) -> None:
        pass
    
        self.nn1 = torch.nn.Sequential(
            
            nn.Conv2d(3, 3, kernel_size=(2,2)),
        )
    
    def step(self, x):
        x = torch.tensor(x).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # add batch and channel dimensions
        x = F.max_pool2d(F.relu(self.nn1(x)), (2,2))
        return x
        
        
neural = NeuralNetwork()

print(neural.step(0.4))
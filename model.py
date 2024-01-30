import torch
import torch.nn as nn
from torchvision import models


class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #############################
        # Initialize your network
        #############################
        
        # Loading the pre-trained GoogLeNet model with default weights
        self.base_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        
        # Extracting the feature extraction layers from the GoogLeNet model
        self.features = nn.ModuleList(self.base_model.children())[:-1]
        self.features = nn.Sequential(*self.features)
        
        # Freezing the parameters of the base model to prevent them from being updated during training
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Getting the number of input features for the fully connected layer
        fc_inputs = self.base_model.fc.in_features
        
        # Defining the architecture of the fully connected layers
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(fc_inputs, 256)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256,8)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        #############################
        # Implement the forward pass
        #############################
        
        x = self.features(x)
        x = self.flat(x)
        x = self.relu1(self.linear1(x))
        x = self.softmax(self.linear2(x))
        return x
    
    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        
        torch.save(self.state_dict(), 'model.pkl')


import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(CNNClassifier, self).__init__()
        # Each input sample is a 2D matrix of shape (128, 6) with 1 channel.
        
        self.dropout_rate = dropout_rate

        # First convolutional layer: (1, 128, 6) → (32, 128, 6)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)

        # Pooling by 2,2 halves the spatial dimensions (32, 128, 6) → (32, 64, 3) 
        self.pool = nn.MaxPool2d((2, 2)) 

        # Second convolutional layer: (32, 64, 3) → (64, 64, 3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)

        # Fully connected layers
        # After two poolings: (64, 32, 1) → flatten into 1d vector of size 2048
        self.fc1 = nn.Linear(64 * 32 * 1, 128)  
        self.fc2 = nn.Linear(128, 12)  # Final output layer, 12 output classes
        
        self.dropout = nn.Dropout(self.dropout_rate) # to prevent overfitting

    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))  # Reshape from original shape (128, 6) to CNN input shape (1, 128, 6) before feeding to the first layer

        x = F.relu(x)
        x = self.pool(x) # -> (batch_size, 32, 64, 3)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x) # -> (batch_size, 64, 32, 1)
        
        x = x.view(-1, 64 * 32 * 1) # flatten by infering automatically the batch size

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x 
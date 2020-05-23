## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        n_img_rows =  n_img_cols = 224
        n_targets= 68*2
        kern1=5
        kern2=3
        kern3=3
        kern4=2
        kern5=1
        n_reduced_rows=( ( ( ( (n_img_rows - (kern1-1))/2 - (kern2-1) )/ 2 - (kern3-1) ) /2 - (kern4-1) ) /2 - (kern5-1) ) / 2
        n_reduced_cols=( ( ( ( (n_img_cols - (kern1-1))/2 - (kern2-1) )/ 2 - (kern3-1) ) /2 - (kern4-1) ) /2 - (kern5-1) ) / 2
        n_features = 512*int(n_reduced_rows) * int(n_reduced_cols)
        self.n_targets = 136
        self.conv1 = nn.Conv2d(1, 32, kern1)
        self.pool1= nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32,64,kern2)
        self.conv3 = nn.Conv2d(64,128,kern3)
        self.conv4 = nn.Conv2d(128,256,kern4) 
        self.conv5 = nn.Conv2d(256,512,kern5)
        self.dropout1 = nn.Dropout(p=0.5)
        self.norm1 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(n_features, 2000)
        self.fc2 = nn.Linear(2000, n_targets)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.conv3(x)))
        x = self.pool1(F.relu(self.conv4(x)))
        x = self.pool1(F.relu(self.conv5(x)))
        x = self.norm1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x

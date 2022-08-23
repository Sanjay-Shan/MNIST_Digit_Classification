import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        #model_2

        #uncommented are the ones used for running the specific mode
        self.conv1=torch.nn.Conv2d(1, 40, kernel_size=5, stride=1, padding=0)  #input image has 1 channel and there are 40 kernels
        self.conv2=torch.nn.Conv2d(40, 60, kernel_size=5, stride=1, padding=0) #output of last layer gives out 40 channels and there are 60 kernels in this layer
        self.fc3=nn.Linear(960,100) # input shape is 28x28 and there are 100 layers in the next layer
        self.fc4=nn.Linear(100,10) # Digit classification has 10 digits in it

        # model_3 requires just the change in the output activation layer

        # # # model_4
        self.fc5=nn.Linear(100,100)

        # model_5
        self.fc6=nn.Linear(960,1000) # input shape is 28x28 and there are 100 layers in the next layer
        self.fc7=nn.Linear(1000,1000) # next layer has 100 neurons
        self.fc8=nn.Linear(1000,10)
        

        #model_1
        self.fc1=nn.Linear(784,100) # input shape is 28x28 and there are 100 layers in the next layer
        self.act=nn.Sigmoid()
        self.fc2=nn.Linear(100,10) # Digit classification has 10 digits in it
        

        #output layer
        self.output=nn.Softmax(dim=1)

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        #forward feed
        x = X.view(X.shape[0], -1)
        x=self.fc1(x)
        x=self.act(x)
        x=self.fc2(x)
        return self.output(x)
    
    def model_2(self, X):
        x=self.conv1(X)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x=self.fc3(x)
        x=self.act(x)
        x=self.fc4(x)
        return self.output(x)

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        x=torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(X)),(2,2))
        x=torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)),(2,2))
        x = x.view(x.shape[0], -1)
        x=torch.nn.functional.relu(self.fc3(x))
        x=torch.nn.functional.relu(self.fc4(x))
        return self.output(x)

    # Add one extra fully connected layer.
    def model_4(self, X):
        x=torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(X)),(2,2))
        x=torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)),(2,2))
        x = x.view(x.shape[0], -1)
        x=torch.nn.functional.relu(self.fc3(x))
        x=torch.nn.functional.relu(self.fc5(x))
        x=torch.nn.functional.relu(self.fc4(x))
        return self.output(x)

    # Use Dropout now.
    def model_5(self, X):
        x=torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(X)),(2,2))
        x=torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)),(2,2))
        x = x.view(x.shape[0], -1)
        x=torch.nn.functional.relu(self.fc6(x))
        x=torch.nn.functional.dropout(x, p=0.5)
        x=torch.nn.functional.relu(self.fc7(x))
        x=torch.nn.functional.dropout(x, p=0.5)
        x=torch.nn.functional.relu(self.fc8(x))
        return self.output(x)
        
    

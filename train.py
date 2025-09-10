import time
import copy
import numpy

import torch
import torch.nn as nn  # neural network building blocks
import torch.optim as optim # model optimizers
import torchvision # for vision models
import torchvision.transforms as transforms # transforming images
from torchvision import datasets, models # cifar-10 dataset and pretrained models
from torch.utils.data import DataLoader

'''
CIFAR-10: dataset of 60000 images with 10 clasess
    - airplanes
    - birds
    - cars
    - cats
    - deers
    - dogs
    - frogs
    - horses
    - ships
    - trucks

train/test split = 50000/10000 --> boolean value preassinged
'''
if torch.cuda.is_available(): # use nvdia gpu (much faster for matrix operations (nn))
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# transform images form CIFAR-10 database for ResNet18 compatability

# ResNet needs 224 x 224 images, and convert images to pytorch tensors
transform = transforms.Compose([transforms.Resize(244), transforms.ToTensor()])

'''
# load CIFAR dataset for training data
train and test flags pre-defined as per dataset

arguments:
root --> where to store dataset
train --> differentiate between training and testing data
download --> straight forward
transform --> apply ResNet transformations

batch_size --> # images per batch
shuffle --> randomize dataset to avoid memorization
num_workers --> how many subprocesses run in parralel and spawned by pytorch

'''
# load CIFAR training set
trainset = datasets.CIFAR10(root="./data", train = True, download = True, transform = transform)
trainloader = DataLoader(trainset, batch_size = 64, shuffle = False, num_workers = 0)

# load CIFAR test set
testset = datasets.CIFAR10(root="./data", train = True, download = True, transform = transform)
testloader = DataLoader(trainset, batch_size = 64, shuffle = False, num_workers = 0)


# load pretrained model for transfer learning as training a convoluted neural network from scratch takes huge compute
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# freeze early layers, we only want to retrained the later more specialized layers (we only need some of the more basic transfer learning)
for param in model.parameters():
    param.requires_grad = False # locks in all pretrained weights

# .fc method retrives last fully conneced layer of the network
num_features = model.fc.in_features # number of last layer features
model.fc = nn.Linear(num_features, 10) # defining ouput layer for 10 classes (that exist in the CIFAR-10 Database)

model = model.to(device) # defining computational home

crtiterion = nn.CrossEntropyLoss() # accuracy measurement for learning (basis for optimization)
optimizer = optim.Adam(model.fc.parameters()) # update weights based on loss gradients

# ~~~ Traning Loop ~~~
epochs = 2 # number of passes through training data -- to be increased if need be

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader: # loop through training data (defined as batchs of 64)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # reset gradients
        outputs = model(images) # pass batch through model
        loss = crtiterion(outputs, labels) # measure accuracy with built in loss function (compare with labels)
        loss.backward() # backpropagate to compute gradients  (tells you how much the loss will change poer tweak)
        optimizer.step() # apply optimization as computed by backpropagation to tune model

        running_loss += loss.item()
    
    # print average loss per epoch
    print(f"Epoch: {epoch + 1} of {epochs}, Loss: {running_loss/len(trainloader):.2f}")

print("Training Done...")

# ~~~ Eval Loop ~~~
correct = 0
total = 0
model.eval() # set model to evaluation mode
with torch.no_grad(): # efficiency (no gradients needed)
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images) # send batch
        values, predicted = torch.max(outputs, 1) # classifies image type --> returns max class score (model confidence per class) and its index

        '''
         a note on torch.max
        | Class 0 | Class 1 | Class 2 | ... | Class 9 |
        | ------- | ------- | ------- | --- | ------- |
        | 1.2     | 0.5     | 2.1     | ... | -0.3    | --> torch max returns 2.1 (highest confidence so predict 2)
        '''
        total += labels.size(0)
        if (predicted == labels):
            correct += 1

        
# Print test accuracy
print(f"Test Accuracy: {100 * correct / total:.2f}%")


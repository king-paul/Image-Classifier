# Imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os

from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image

# CLASSES #
class Network(nn.Module):
    '''feed-forward network as a classifier, using ReLU activations and dropout '''
    def __init__(self, input_units, h1_units, h2_units, output_units):
        super().__init__()
        # Defining the layers
        self.fc1 = nn.Linear(input_units, h1_units)        
        # Inputs to hidden layer linear transformation
        self.fc2 = nn.Linear(h1_units, h2_units)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(h2_units, output_units)
        
    def forward(self, x):        
        ''' Forward pass through the network, returns the output logits '''
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.output(x)
        x = F.softmax(x, dim=1)
        
        return x
    
# FUNCTIONS #

def load_data(data_dir):
    ''' 
    Loads the tesing, training and valid image data from a specified directory then then creates data sets
    frim them using composed transforms. Data loaders are then created using the datasets which are then
    returned from this function.
    '''
    
    # define paths to folders
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'    
    
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.Resize((224, 224)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_set = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_set  = datasets.ImageFolder(test_dir , transform= test_transforms)
    valid_set = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Todo: Using the image datasets and the trainforms, define the dataloaders
    #dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=4, shuffle=True)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    testloader  = torch.utils.data.DataLoader(test_set , batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=32, shuffle=True)
    
    return trainloader, testloader, validloader

def train_model(trainloader, testloader, model, learning_rate = 0.001, epochs = 1):
    ''' This is the method used to execute the training process after data has been loaded and everything else has been defined ''' 
    epochs = command_args.epochs # total iterations
    steps = 0 # number of training steps performed
    running_loss = 0 
    print_every = 5 # number of steps before print out the validation loss
    
    print("\nPreparing to train model...")    
    # set up device to be the GPU if cude is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.NLLLoss() # Natural log loss
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print("Device Type Used:", device.type)
    model.to(device)
    
    print("Training the model...") 
    for epoch in range(epochs):
        print("\nBeginning Epoch", epoch)  
        # increment steps every time a batch has been processed
        for images, labels, in trainloader:        
            steps += 1
            print("step", steps)
            
            #images.unsqueeze(0)
            # move images and labels over to GPU (if available)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() # zeroes out all the gradients   

            logps = model(images) # gets log probabilites from the model
            loss = criterion(logps, labels) # gets loss from criterion using log probabilities
            loss.backward() # backwards pass
            optimizer.step()

            running_loss += loss.item() # increments running loss by loss from current batch

            # test network accuracy and loss on test data set
            if steps % print_every == 0:
                model.eval() # turns on evaluation inference mode (will turn of dropout)
                test_loss = 0
                accuracy = 0

                # iterates through all images and labels in the test data
                for images, labels in testloader:

                    images, labels = images.to(device), labels.to(device) # transfera tensors to gpu

                    logps = model(images)
                    loss = criterion(logps, labels)
                    test_loss += loss.item() # keeps trach of loss when going through validation loops

                    # calculate our accuracy
                    ps = torch.exp(logps) # uses exponential to conver log probabilites to actual probabilites
                    top_ps, top_class = ps.topk(1, dim=1) # returns first largest value in in probanilities from first row
                    equality = top_class == labels.view(*top_class.shape) # checks for equality using labels
                    accuracy += torch.mean(equality.type(torch.FloatTensor))

                # print out info
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. " # avarage of training loss
                      f"Test loss: {test_loss/len(testloader):.3f}.. " # average loss. len(testloader) = number of batches
                      f"Test accuracy: {accuracy/len(testloader):.3f}") # avarage accuaracy

                # reset running loss and return model to training mode
                running_loss = 0
                model.train()
            
            print("step complete")

def save_checkpoint(save_dir):
    torch.save(model.state_dict(), save_dir + '/flower_classification_model.pth')
    print("\nThe trained data has been saved to", save_dir + '/flower_classification_model.pth')
    
def parse_arguments():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # parse the different types of arguments
    parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'Path to directory containing the training and test data')
    parser.add_argument('--save_dir', type = str, default = '.', help = 'Path to save the checkpoints')
    parser.add_argument('--arch', type = str, default = "vgg13", help = 'The architecture used to train the model with')
    parser.add_argument('--learning_rate', type = float, default = 0.01)
    parser.add_argument('--hidden_units', type = int, default = 256)
    parser.add_argument('--epochs', type = int, default = 20)

    # Return parsed argument collection
    return parser.parse_args()
    
# START OF PROGRAM #
if __name__ == "__main__":
    
    command_args = parse_arguments()
    
    # print out the command line arguments
    print("Model Training settings:")
    print(f'Image Data Folder: {os.getcwd()}/{command_args.data_dir}')
    print(f'Save Directory: {os.getcwd()}/{command_args.save_dir}')
    print('Architecture Used:', command_args.arch)
    print('Learning Rate:', command_args.learning_rate)
    print('Hidden Units:', command_args.hidden_units)
    print('Epochs:', command_args.epochs)
    
    load_data(command_args.data_dir)

    # define classifier
    classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=command_args.hidden_units, bias=True),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(in_features=command_args.hidden_units, out_features=102, bias=True),
                               nn.LogSoftmax(dim=1))
    
    # Build a feed-forward network
    model = models.vgg11(pretrained=True)
    #model = Network(224, 43008, 64, 10)
    model.classifier = classifier # attaches classifier to model
    
    # load image data from folders
    trainloader, testloader, validloader = load_data(command_args.data_dir)

    # train the model    
    train_model(trainloader, testloader, model)

    # save the model
    save_checkpoint(command_args.save_dir)